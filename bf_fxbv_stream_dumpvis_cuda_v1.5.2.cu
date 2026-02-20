// bf_fxbv_stream_dumpvis_cuda_v1.5.0.cu
////lijian@xao.ac.cn #20260220
// CUDA version of bf_fxbv_stream_dumpvis (PSRDADA 4-in -> 1-out FXB(+V) beamformer)
//   - Uses cuFFT for FFT<N> / IFFT<N> (runtime selectable via --nfft)
//   - Adds --cuda <device_id> to select GPU
//   - Batch processing (default --batch-frames 256) to avoid per-frame overhead
//   - VISDUMP output compatible with plot_visdump_fringe_v2.py
//
// Input ring payload layout (from your udp2dada_vdif_blk2_* receiver):
//   Xre[1024], Xim[1024], Yre[1024], Yim[1024]  (int16), total 8192 bytes per "frame"
//
// Output ring payload layout (same):
//   Xre[1024], Xim[1024], Yre[1024], Yim[1024]  (int16), total 8192 bytes per "frame"
//
// Build example (use SAME PSRDADA include/lib as udp2dada):
//   nvcc -O3 -std=c++14 -arch=sm_86 -o bf_fxbv_stream_dumpvis_cuda bf_fxbv_stream_dumpvis_cuda.cu \
//     -lcufft -lm -lpthread \
//     -I /home/uwb/linux_64/include \
//     -L /home/uwb/linux_64/lib -Wl,-rpath,/home/uwb/linux_64/lib -lpsrdada
//
// Run example:
//   dada_dbnull -k b000 &
//   ./bf_fxbv_stream_dumpvis_cuda --cuda 0 --batch-frames 256 \
//      --in0 a000 --in1 a002 --in2 a004 --in3 a006 --out b000 \
//      --fs 32e6 --monitor 1 --monitor-sec 0.1 --dump-vis vis01.bin \
//      --dump-baseline 0-1 --dump-pol XX --no-delay 1
//
// Notes:
//   - Inputs are SINGLE-READER: stop any dada_dbmonitor/dbnull on input rings.
//   - Output ring needs a consumer (dada_dbnull -k b000), otherwise writer will block.
//   - This program follows your PSRDADA behavior: NO ipcio_open(); uses ipcio_read/ipcio_write directly.
//
// 2026-02-18

#define BF_CUDA_VERSION "v1.5.2"


#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <strings.h>
#include <sched.h>

#include <cuda_runtime.h>
#include <cufft.h>

// psrdada headers are C
extern "C" {
#include <dada_hdu.h>
#include <ipcio.h>
#include <ipcbuf.h>
#include <ascii_header.h>
#include <multilog.h>
}

// SOFA (IAU) C library for RA/Dec -> Az/El
extern "C" {
#include "sofa.h"
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define NANT 4
#define NPOL 2

#define FRAME_SAMP 1024
#define NFFT_DEFAULT 32
#define NFFT_MAX 1024
// NOTE: nspec is runtime: nspec = FRAME_SAMP / nfft

#define PAYLOAD_SIZE 8192
#define SHORTS_PER_FRAME (PAYLOAD_SIZE/2)      // 4096 int16
#define BLOCK_LEN 1024

static volatile int g_stop = 0;
static void on_sigint(int){ g_stop = 1; }

static inline double now_sec(){
  timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9;
}

static key_t parse_key(const char* s)
{
  char* e = nullptr;
  unsigned long v = strtoul(s, &e, 16);
  if(e && *e) v = strtoul(s, nullptr, 10);
  return (key_t)v;
}


// Bind the *process/threads* to a single CPU core (Linux).
// New threads created after this call inherit the affinity mask.
static int bind_cpu_core(int core_id)
{
  if(core_id < 0) return 0;

#ifdef __linux__
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if(ncpu > 0 && core_id >= ncpu){
    fprintf(stderr,
            "[bf-cuda %s][WARN] --cpu-core=%d out of range (0..%ld); not binding\n",
            BF_CUDA_VERSION, core_id, ncpu-1);
    return -1;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core_id, &set);

  if(sched_setaffinity(0, sizeof(set), &set) != 0){
    fprintf(stderr,
            "[bf-cuda %s][WARN] sched_setaffinity(core=%d) failed: %s\n",
            BF_CUDA_VERSION, core_id, strerror(errno));
    return -1;
  }

  // Verify
  cpu_set_t get;
  CPU_ZERO(&get);
  if(sched_getaffinity(0, sizeof(get), &get) == 0){
    int ok = CPU_ISSET(core_id, &get) ? 1 : 0;
    fprintf(stderr, "[bf-cuda %s] cpu affinity set to core=%d (verify=%d)\n",
            BF_CUDA_VERSION, core_id, ok);
  } else {
    fprintf(stderr, "[bf-cuda %s] cpu affinity set to core=%d\n",
            BF_CUDA_VERSION, core_id);
  }
  return 0;
#else
  (void)core_id;
  fprintf(stderr, "[bf-cuda %s][WARN] --cpu-core is only supported on Linux\n",
          BF_CUDA_VERSION);
  return -1;
#endif
}

#define CUDA_OK(stmt) do{ cudaError_t _e=(stmt); if(_e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} }while(0)

#define CUFFT_OK(stmt) do{ cufftResult _r=(stmt); if(_r!=CUFFT_SUCCESS){ \
  fprintf(stderr,"[CUFFT] %s:%d: error code %d\n", __FILE__, __LINE__, (int)_r); exit(1);} }while(0)

// ------------------- Geometry (ENU; can be loaded from file at runtime) -------------------
// Default: the original demo ENU (meters) used in earlier versions.
static double ant_enu[NANT][3] = {
  {   0.0,   0.0, 0.0 },
  { -300.0,  0.0, 0.0 },
  {   0.0, 250.0, 0.0 },
  {   0.0, 300.0, 0.0 },
};

// Geodetic positions (from --antfile). Needed for RA/Dec -> Az/El via SOFA.
static double ant_geo_lon_deg[NANT] = {0};
static double ant_geo_lat_deg[NANT] = {0};
static double ant_geo_h_m[NANT] = {0};
static int ant_geo_valid = 0;

// WGS84 constants for GEO->ECEF->ENU
static const double WGS84_A  = 6378137.0;                 // semi-major axis (m)
static const double WGS84_F  = 1.0 / 298.257223563;       // flattening
static const double WGS84_E2 = WGS84_F * (2.0 - WGS84_F); // eccentricity^2

static inline double deg2rad(double deg){ return deg * M_PI / 180.0; }

static void geodetic_to_ecef(double lat_deg, double lon_deg, double h_m, double xyz[3])
{
  const double lat = deg2rad(lat_deg);
  const double lon = deg2rad(lon_deg);
  const double sin_lat = sin(lat);
  const double cos_lat = cos(lat);
  const double sin_lon = sin(lon);
  const double cos_lon = cos(lon);

  const double N = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);

  xyz[0] = (N + h_m) * cos_lat * cos_lon;
  xyz[1] = (N + h_m) * cos_lat * sin_lon;
  xyz[2] = (N * (1.0 - WGS84_E2) + h_m) * sin_lat;
}

static void ecef_to_enu(const double dxyz[3], double lat0_deg, double lon0_deg, double enu[3])
{
  const double lat0 = deg2rad(lat0_deg);
  const double lon0 = deg2rad(lon0_deg);
  const double sin_lat = sin(lat0);
  const double cos_lat = cos(lat0);
  const double sin_lon = sin(lon0);
  const double cos_lon = cos(lon0);

  // E
  enu[0] = -sin_lon * dxyz[0] + cos_lon * dxyz[1];
  // N
  enu[1] = -sin_lat * cos_lon * dxyz[0] - sin_lat * sin_lon * dxyz[1] + cos_lat * dxyz[2];
  // U
  enu[2] =  cos_lat * cos_lon * dxyz[0] + cos_lat * sin_lon * dxyz[1] + sin_lat * dxyz[2];
}

// Load antenna geodetic coordinates from a CSV-like text file and compute ENU relative to a reference antenna.
// File format (ignore lines starting with # or empty):
//   id, lon_deg, lat_deg, elev_m
// Example:
//   0,87.18,43.47,2080.0
//   1,87.18,43.50,2081.0
//   2,87.19,43.49,2082.0
//   3,87.20,43.49,2083.0
//
// This updates the global ant_enu[][] array.
static bool load_antfile_geo_csv(const char* path, int ref_ant)
{
  struct Entry { bool ok; double lon, lat, h; };
  Entry ent[NANT];
  for(int a=0;a<NANT;a++){ ent[a].ok=false; ent[a].lon=0; ent[a].lat=0; ent[a].h=0; }

  FILE* fp = fopen(path, "r");
  if(!fp){
    fprintf(stderr, "[bf-cuda %s][ERR] cannot open --antfile %s: %s\n", BF_CUDA_VERSION, path, strerror(errno));
    return false;
  }

  char line[256];
  while(fgets(line, (int)sizeof(line), fp)){
    char* s = line;
    while(*s==' ' || *s=='\t') s++;
    if(*s=='#' || *s=='\n' || *s==0) continue;

    // replace commas with spaces to make sscanf easier
    for(char* p=s; *p; ++p) if(*p==',') *p=' ';

    int id;
    double lon_deg, lat_deg, h_m;
    if(sscanf(s, "%d %lf %lf %lf", &id, &lon_deg, &lat_deg, &h_m) == 4){
      if(id < 0 || id >= NANT){
        fprintf(stderr, "[bf-cuda %s][WARN] antfile: skip id=%d (expect 0..%d)\n", BF_CUDA_VERSION, id, NANT-1);
        continue;
      }
      ent[id].ok  = true;
      ent[id].lon = lon_deg;
      ent[id].lat = lat_deg;
      ent[id].h   = h_m;
    }
  }
  fclose(fp);

  for(int a=0;a<NANT;a++){
    if(!ent[a].ok){
      fprintf(stderr, "[bf-cuda %s][ERR] antfile missing antenna id=%d\n", BF_CUDA_VERSION, a);
      return false;
    }
  }
  if(ref_ant < 0 || ref_ant >= NANT){
    fprintf(stderr, "[bf-cuda %s][ERR] --ref-ant must be 0..%d\n", BF_CUDA_VERSION, NANT-1);
    return false;
  }

  // cache geodetic coordinates for SOFA pointing (RA/Dec -> Az/El)
  for(int a=0;a<NANT;a++){
    ant_geo_lon_deg[a] = ent[a].lon;
    ant_geo_lat_deg[a] = ent[a].lat;
    ant_geo_h_m[a]     = ent[a].h;
  }
  ant_geo_valid = 1;

  // GEO -> ECEF
  double ecef[NANT][3];
  for(int a=0;a<NANT;a++){
    double xyz[3];
    geodetic_to_ecef(ent[a].lat, ent[a].lon, ent[a].h, xyz); // lat, lon order!
    ecef[a][0]=xyz[0]; ecef[a][1]=xyz[1]; ecef[a][2]=xyz[2];
  }

  const double lat0 = ent[ref_ant].lat;
  const double lon0 = ent[ref_ant].lon;
  const double refxyz[3] = { ecef[ref_ant][0], ecef[ref_ant][1], ecef[ref_ant][2] };

  // ECEF -> ENU (relative)
  for(int a=0;a<NANT;a++){
    const double dxyz[3] = { ecef[a][0]-refxyz[0], ecef[a][1]-refxyz[1], ecef[a][2]-refxyz[2] };
    double enu[3];
    ecef_to_enu(dxyz, lat0, lon0, enu);
    ant_enu[a][0]=enu[0]; ant_enu[a][1]=enu[1]; ant_enu[a][2]=enu[2];
  }

  fprintf(stderr, "[bf-cuda %s] Loaded antenna GEO from %s (ref_ant=%d)\n", BF_CUDA_VERSION, path, ref_ant);
  for(int a=0;a<NANT;a++){
    fprintf(stderr, "[bf-cuda %s] ant%d GEO lon=%.6f lat=%.6f h=%.3f m  -> ENU E=%.3f N=%.3f U=%.3f m\n",
            BF_CUDA_VERSION, a, ent[a].lon, ent[a].lat, ent[a].h,
            ant_enu[a][0], ant_enu[a][1], ant_enu[a][2]);
  }

  return true;
}
static const double C_LIGHT = 299792458.0;

static void unitvec_enu(double az_deg, double el_deg, double out[3]){
  double az = az_deg * M_PI/180.0;
  double el = el_deg * M_PI/180.0;
  out[0] = cos(el)*sin(az); // E
  out[1] = cos(el)*cos(az); // N
  out[2] = sin(el);         // U
}

static void geometric_delays(double az_deg, double el_deg, double tau[NANT]){
  double s[3]; unitvec_enu(az_deg, el_deg, s);
  double t0 = 0.0;
  for(int a=0;a<NANT;a++){
    double dot = ant_enu[a][0]*s[0] + ant_enu[a][1]*s[1] + ant_enu[a][2]*s[2];
    tau[a] = -dot / C_LIGHT;
    if(a==0) t0 = tau[a];
  }
  for(int a=0;a<NANT;a++) tau[a] -= t0;
}


// ------------------- SOFA pointing: ICRS RA/Dec -> observed Az/El -------------------
// If --ra/--dec are provided, the beamformer can compute Az/El automatically from
// the observation time (prefer UTC_START from PSRDADA header, otherwise system UTC)
// and update delay/fringe-stopping weights every --track-sec.

#define RAD2DEG (180.0/M_PI)

static bool parse_ra_hms_sofa(const char* s, double* ra_rad){
  int h=0, m=0; double sec=0.0;
  if(sscanf(s, "%d:%d:%lf", &h, &m, &sec) != 3 &&
     sscanf(s, "%d %d %lf", &h, &m, &sec) != 3){
    return false;
  }
  // SOFA: 24h -> 2pi
  int st = iauTf2a('+', h, m, sec, ra_rad);
  return (st == 0);
}

static bool parse_dec_dms_sofa(const char* s, double* dec_rad){
  char sign = '+';
  int d=0, m=0; double sec=0.0;

  if(sscanf(s, " %c%d:%d:%lf", &sign, &d, &m, &sec) == 4 ||
     sscanf(s, " %c%d %d %lf",  &sign, &d, &m, &sec) == 4){
    if(sign != '+' && sign != '-') sign = '+';
    int st = iauAf2a(sign, d, m, sec, dec_rad);
    return (st == 0);
  }

  // without sign => '+'
  if(sscanf(s, "%d:%d:%lf", &d, &m, &sec) == 3 ||
     sscanf(s, "%d %d %lf",  &d, &m, &sec) == 3){
    int st = iauAf2a('+', d, m, sec, dec_rad);
    return (st == 0);
  }
  return false;
}

static void get_system_utc_ymdhms(int* y,int* mo,int* d,int* hh,int* mm,double* ss){
  struct timeval tv; gettimeofday(&tv, NULL);
  time_t t = tv.tv_sec;
  struct tm g;
  gmtime_r(&t, &g);
  *y  = g.tm_year + 1900;
  *mo = g.tm_mon + 1;
  *d  = g.tm_mday;
  *hh = g.tm_hour;
  *mm = g.tm_min;
  *ss = (double)g.tm_sec + (double)tv.tv_usec * 1e-6;
}

static bool parse_utc_ymdhms(const char* s, int* y,int* mo,int* d,int* hh,int* mm,double* ss){
  if(!s || !s[0]) return false;
  // Accept:
  // YYYY-MM-DDTHH:MM:SS(.sss)
  // YYYY-MM-DD HH:MM:SS(.sss)
  // YYYY-MM-DD-HH:MM:SS(.sss)   (common in PSRDADA headers)
  if(sscanf(s, "%d-%d-%dT%d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return true;
  if(sscanf(s, "%d-%d-%d %d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return true;
  if(sscanf(s, "%d-%d-%d-%d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return true;
  return false;
}

static bool sofa_utc_to_jd(int y,int mo,int d,int hh,int mm,double ss, double* utc1, double* utc2){
  int st = iauDtf2d("UTC", y, mo, d, hh, mm, ss, utc1, utc2);
  return (st == 0);
}

// Format a UTC Julian Date (2-part) into ISO string YYYY-MM-DDTHH:MM:SS.sss
static bool sofa_jd_to_utc_iso(double utc1, double utc2, char* out, size_t outsz){
  if(!out || outsz < 8) return false;
  int iy=0, im=0, id=0;
  int ihmsf[4] = {0,0,0,0};
  int ndp = 3; // milliseconds
  int st = iauD2dtf("UTC", ndp, utc1, utc2, &iy, &im, &id, ihmsf);
  if(st != 0) return false;
  // ihmsf: hour, min, sec, fraction(10^-ndp)
  snprintf(out, outsz, "%04d-%02d-%02dT%02d:%02d:%02d.%0*d",
           iy, im, id, ihmsf[0], ihmsf[1], ihmsf[2], ndp, ihmsf[3]);
  return true;
}

static bool sofa_radec_to_azel_utc(double ra_rad, double dec_rad,
                                  double utc1, double utc2, double dut1,
                                  double lon_deg, double lat_deg, double hm_m,
                                  double xp_rad, double yp_rad,
                                  double phpa, double tc, double rh, double wl,
                                  double* az_deg, double* el_deg)
{
  // Note: SOFA expects longitude East-positive in radians.
  double lon = deg2rad(lon_deg);
  double lat = deg2rad(lat_deg);

  double aob=0.0, zob=0.0, hob=0.0, dob=0.0, rob=0.0, eo=0.0;
  iauAtco13(ra_rad, dec_rad,
            0.0, 0.0, 0.0, 0.0,         // pmRA, pmDec, parallax, rv
            utc1, utc2, dut1,
            lon, lat, hm_m,
            xp_rad, yp_rad,
            phpa, tc, rh, wl,
            &aob, &zob, &hob, &dob, &rob, &eo);

  double az = fmod(aob * RAD2DEG + 360.0, 360.0);
  double el = (M_PI/2.0 - zob) * RAD2DEG;
  *az_deg = az;
  *el_deg = el;
  return true;
}

static bool sofa_radec_to_azel_all(double ra_rad, double dec_rad,
                                  double utc1, double utc2, double dut1,
                                  double xp_rad, double yp_rad,
                                  double phpa, double tc, double rh, double wl,
                                  double az_deg_out[NANT], double el_deg_out[NANT])
{
  if(!ant_geo_valid){
    return false;
  }
  for(int a=0;a<NANT;a++){
    double az=0.0, el=0.0;
    if(!sofa_radec_to_azel_utc(ra_rad, dec_rad, utc1, utc2, dut1,
                              ant_geo_lon_deg[a], ant_geo_lat_deg[a], ant_geo_h_m[a],
                              xp_rad, yp_rad, phpa, tc, rh, wl,
                              &az, &el)){
      return false;
    }
    az_deg_out[a] = az;
    el_deg_out[a] = el;
  }
  return true;
}


// ------------------- VISDUMP format (binary; record length depends on nfft) -------------------
typedef struct __attribute__((packed)) {
  char magic[8];        // "VISDUMP1"
  uint32_t nfft;        // FFT length used for visibilities
  double fs_hz;
  double monitor_sec;   // nominal monitor integration
  uint32_t bi, bj;
  uint32_t pol_code;    // 0=XX 1=YY 2=XY 3=YX
  uint32_t reserved;
  uint8_t pad[64-44];
} visdump_hdr_t;

// Host-side complex float used by VISDUMP and accumulators.
// Must be layout-compatible with CUDA's float2 (x=re, y=im).
typedef struct { float re, im; } cf32;

static void write_visdump_header(FILE* fp, double fs_hz, double monitor_sec,
                                 uint32_t bi, uint32_t bj, uint32_t pol,
                                 uint32_t nfft)
{
  visdump_hdr_t h;
  memset(&h, 0, sizeof(h));
  memcpy(h.magic, "VISDUMP1", 8);
  h.nfft = nfft;
  h.fs_hz = fs_hz;
  h.monitor_sec = monitor_sec;
  h.bi = bi;
  h.bj = bj;
  h.pol_code = pol;
  h.reserved = 0;
  fwrite(&h, 1, sizeof(h), fp);
  fflush(fp);
}

static void write_visdump_record(FILE* fp, double t_mono, float coh_avg,
                                 const cf32* vis, const double* p0, const double* p1,
                                 uint32_t nfft)
{
  // Record layout (variable-sized):
  //   double t_mono; float coh; float pad;
  //   cf32 vis[nfft]; double p0[nfft]; double p1[nfft]
  float pad0 = 0.0f;
  fwrite(&t_mono, sizeof(double), 1, fp);
  fwrite(&coh_avg, sizeof(float), 1, fp);
  fwrite(&pad0, sizeof(float), 1, fp);
  fwrite(vis, sizeof(cf32), nfft, fp);
  fwrite(p0, sizeof(double), nfft, fp);
  fwrite(p1, sizeof(double), nfft, fp);
  fflush(fp);
}

// ------------------- read/write helper (no ipcio_open) -------------------
static int read_exact(ipcio_t* ipc, char* buf, size_t bytes){
  size_t got = 0;
  while(got < bytes){
    ssize_t r = ipcio_read(ipc, buf+got, bytes-got);
    if(r < 0) return -1;
    if(r == 0) return 0; // EOD
    got += (size_t)r;
  }
  return 1;
}

static int write_exact(ipcio_t* ipc, const char* buf, size_t bytes){
  size_t put = 0;
  while(put < bytes){
    ssize_t w = ipcio_write(ipc, (char*)(buf+put), bytes-put);
    if(w < 0) return -1;
    if(w == 0) return 0;
    put += (size_t)w;
  }
  return 1;
}

// ------------------- CLI helpers -------------------
static bool parse_baseline(const char* s, int& bi, int& bj){
  if(!s) return false;
  int a=-1,b=-1;
  if(sscanf(s, "%d-%d", &a, &b) == 2 || sscanf(s, "%d,%d", &a, &b) == 2 || sscanf(s, "%d:%d", &a, &b)==2){
    bi=a; bj=b; return true;
  }
  return false;
}

static int pol_code_from_str(const char* s){
  if(!s) return -1;
  if(!strcasecmp(s,"XX")) return 0;
  if(!strcasecmp(s,"YY")) return 1;
  if(!strcasecmp(s,"XY")) return 2;
  if(!strcasecmp(s,"YX")) return 3;
  return -1;
}


static bool parse_antmask(const char* s, int mask[NANT]){
  if(!s) return false;
  std::string t(s);
  // remove whitespace
  t.erase(std::remove_if(t.begin(), t.end(), [](unsigned char c){ return std::isspace(c); }), t.end());
  if(t.empty()) return false;

  // strip brackets if present
  if(!t.empty() && (t.front()=='[' || t.front()=='(' || t.front()=='{')) t.erase(t.begin());
  if(!t.empty() && (t.back()==']' || t.back()==')' || t.back()=='}')) t.pop_back();
  if(t.empty()) return false;

  std::vector<int> vals;
  // split by comma if present
  if(t.find(',') != std::string::npos){
    size_t start=0;
    while(start < t.size()){
      size_t comma = t.find(',', start);
      std::string part = (comma==std::string::npos) ? t.substr(start) : t.substr(start, comma-start);
      if(part.empty()) return false;
      if(part!="0" && part!="1") return false;
      vals.push_back(part=="1" ? 1 : 0);
      if(comma==std::string::npos) break;
      start = comma + 1;
    }
  } else {
    // accept "1110" form
    if((int)t.size() != NANT) return false;
    for(char c: t){
      if(c!='0' && c!='1') return false;
      vals.push_back(c=='1' ? 1 : 0);
    }
  }

  if((int)vals.size() != NANT) return false;
  for(int i=0;i<NANT;i++) mask[i]=vals[i];
  return true;
}

static void antmask_to_str(const int mask[NANT], char out[32]){
  snprintf(out, 32, "%d,%d,%d,%d", mask[0], mask[1], mask[2], mask[3]);
}


// Map pol_code to per-antenna polarization indices (NPOL=2: 0=X, 1=Y)
static inline void polcode_to_pab(int pol_code, int &pA, int &pB){
  if(pol_code == 0){ pA=0; pB=0; }       // XX
  else if(pol_code == 1){ pA=1; pB=1; }  // YY
  else if(pol_code == 2){ pA=0; pB=1; }  // XY
  else if(pol_code == 3){ pA=1; pB=0; }  // YX
  else { pA=0; pB=0; }
}

static inline const char* polcode_to_str(int pol_code){
  if(pol_code == 0) return "XX";
  if(pol_code == 1) return "YY";
  if(pol_code == 2) return "XY";
  if(pol_code == 3) return "YX";
  return "UN";
}

// Parse a comma-separated list like "XX,YY" into vector of pol_code values.
// Accepts whitespace and optional brackets.
static bool parse_pol_list(const char* s, std::vector<int> &out){
  out.clear();
  if(!s) return false;
  std::string t(s);
  t.erase(std::remove_if(t.begin(), t.end(), [](unsigned char c){ return std::isspace(c); }), t.end());
  if(t.empty()) return false;
  if(!t.empty() && (t.front()=='[' || t.front()=='(' || t.front()=='{')) t.erase(t.begin());
  if(!t.empty() && (t.back()==']' || t.back()==')' || t.back()=='}')) t.pop_back();
  if(t.empty()) return false;

  size_t start=0;
  while(start < t.size()){
    size_t comma = t.find(',', start);
    std::string part = (comma==std::string::npos) ? t.substr(start) : t.substr(start, comma-start);
    if(part.empty()) return false;
    int pc = pol_code_from_str(part.c_str());
    if(pc < 0) return false;
    out.push_back(pc);
    if(comma==std::string::npos) break;
    start = comma + 1;
  }
  // de-dup
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return !out.empty();
}

static bool ensure_dir_exists(const char* dir){
  if(!dir || !dir[0]) return false;
  struct stat st;
  if(stat(dir, &st) == 0){
    if(S_ISDIR(st.st_mode)) return true;
    fprintf(stderr,"[ERR] %s exists but is not a directory\n", dir);
    return false;
  }
  // create (single level)
  if(mkdir(dir, 0755) == 0) return true;
  if(errno == EEXIST) return true;
  fprintf(stderr,"[ERR] mkdir(%s) failed: %s\n", dir, strerror(errno));
  return false;
}

// Load per-antenna per-pol per-channel complex weights from phasecal_weights_v1.csv
// Columns (CSV): ant_id, pol_idx(0=X,1=Y), chan_k, freq_hz, w_re, w_im
// Any missing entries remain 1+0j. Ant0 is forced to 1+0j (reference).
// Load per-antenna per-pol per-channel complex weights from phasecal_weights_v1.csv
// Columns (CSV): ant_id, pol_idx(0=X,1=Y), chan_k, freq_hz, w_re, w_im
// Any missing entries remain 1+0j. Ant0 is forced to 1+0j (reference).
static bool load_calfile_csv(const char* path, int nfft, double fs_hz, std::vector<float2> &cal){
  cal.assign((size_t)NANT * (size_t)NPOL * (size_t)nfft, make_float2(1.0f, 0.0f));
  if(!path) return true; // no file -> unity

  std::ifstream in(path);
  if(!in.is_open()){
    fprintf(stderr,"[ERR] cannot open --calfile %s\n", path);
    return false;
  }

  auto trim = [](std::string &s){
    size_t p = 0;
    while(p < s.size() && std::isspace((unsigned char)s[p])) p++;
    s.erase(0, p);
    while(!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
  };

  int file_nfft = -1;
  double file_fs = 0.0;

  std::string line;
  int nload = 0;
  int nskip = 0;

  while(std::getline(in, line)){
    if(line.empty()) continue;

    // comment/header lines: try to parse metadata like "# nfft=32"
    if(line[0] == '#'){
      size_t p1 = line.find("nfft=");
      if(p1 != std::string::npos){
        file_nfft = atoi(line.c_str() + p1 + 5);
      }
      size_t p2 = line.find("fs_hz=");
      if(p2 != std::string::npos){
        file_fs = atof(line.c_str() + p2 + 6);
      }
      continue;
    }

    std::stringstream ss(line);
    std::string f0,f1,f2,f3,f4,f5;
    if(!std::getline(ss,f0,',')) continue;
    if(!std::getline(ss,f1,',')) continue;
    if(!std::getline(ss,f2,',')) continue;
    if(!std::getline(ss,f3,',')) continue;
    if(!std::getline(ss,f4,',')) continue;
    if(!std::getline(ss,f5,',')) continue;
    trim(f0); trim(f1); trim(f2); trim(f4); trim(f5);

    int ant = atoi(f0.c_str());
    int pol = atoi(f1.c_str());
    int k   = atoi(f2.c_str());
    float wr = (float)atof(f4.c_str());
    float wi = (float)atof(f5.c_str());
    if(ant < 0 || ant >= NANT){ nskip++; continue; }
    if(pol < 0 || pol >= NPOL){ nskip++; continue; }
    if(k < 0 || k >= nfft){ nskip++; continue; }

    cal[((size_t)ant * (size_t)NPOL + (size_t)pol) * (size_t)nfft + (size_t)k] = make_float2(wr, wi);
    nload++;
  }

  if(file_nfft > 0 && file_nfft != nfft){
    fprintf(stderr,"[bf-cuda %s][ERR] calfile nfft=%d does not match --nfft=%d (%s)\n",
            BF_CUDA_VERSION, file_nfft, nfft, path);
    return false;
  }
  if(file_fs > 0.0 && fs_hz > 0.0){
    double rel = fabs(file_fs - fs_hz) / fs_hz;
    if(rel > 1e-6){
      fprintf(stderr,"[bf-cuda %s][WARN] calfile fs_hz=%.6f differs from --fs=%.6f (rel=%.3g). Proceeding.\n",
              BF_CUDA_VERSION, file_fs, fs_hz, rel);
    }
  }

  // Force reference antenna to unity
  for(int pol=0; pol<NPOL; pol++){
    for(int k=0; k<nfft; k++){
      cal[((size_t)0 * (size_t)NPOL + (size_t)pol) * (size_t)nfft + (size_t)k] = make_float2(1.0f, 0.0f);
    }
  }

  fprintf(stderr,"[bf-cuda %s] loaded cal weights: %s (rows=%d, skipped=%d)\n", BF_CUDA_VERSION, path, nload, nskip);
  return true;
}

typedef struct {
  int bi, bj;
  int pol_code;
  int pA, pB;
  FILE* fp;  // may be NULL for monitor-only target
  std::vector<cf32> vis_acc;
  std::vector<double> p0_acc;
  std::vector<double> p1_acc;
  uint64_t frames_acc;
  // last monitor snapshot (updated every monitor tick)
  int have_last;
  float last_coh_avg;
  float last_snr_rad;
  float last_snr_emp;
  double last_bw_eff_hz;
  double last_tint_emp;
  double last_tau_res;
  double last_p0_mean;
  double last_p1_mean;
  double last_t_mon;
} dump_target_t;

static dump_target_t* find_target(std::vector<dump_target_t>& v, int bi, int bj, int pol_code){
  for(auto &t : v){
    if(t.bi==bi && t.bj==bj && t.pol_code==pol_code) return &t;
  }
  for(auto &t : v){
    if(t.bi==bj && t.bj==bi && t.pol_code==pol_code) return &t;
  }
  return nullptr;
}

static void usage(const char* prog){
  fprintf(stderr,
    "Usage: %s --cuda <id> [options]\n"
    "\n"
    "Inputs/Outputs:\n"
    "  --in0 <key> --in1 <key> --in2 <key> --in3 <key>   PSRDADA input keys (default a000,a002,a004,a006)\n"
    "  --out <key>                                      PSRDADA output key (default b000)\n"
    "\n"
    "Signal:\n"
    "  --fs <Hz>              sampling rate (default 32e6)\n"
    "  --fcenter-hz <Hz>      RF center frequency used for carrier/fringe phase (default 1.42e9)\n"
    "  --nfft <N>             FFT length: 32/64/128/256/512/1024 (default 32)\n"
    "\n"
    "Pointing / tracking:\n"
    "  --az <deg> --el <deg>  fixed az/el (default az=0 el=90)\n"
    "  --ra \"HH:MM:SS.S\" --dec \"+DD:MM:SS.S\"   track ICRS RA/Dec using SOFA (requires --antfile)\n"
    "  --track-sec <sec>      update delays every this many seconds of data-time (default 1)\n"
    "  --track-csv <file>     append tracking CSV rows each update (UTC, Az/El, tau, coarse/fine, coh/power, snr)\n"
    "  --snr-mode <rad|emp>   tracking CSV SNR estimator (default emp)\n"
    "\n"
    "SOFA / EOP / refraction (optional):\n"
    "  --dut1 <sec>  --xp <arcsec>  --yp <arcsec>\n"
    "  --phpa <hPa>  --tc <C>       --rh <0..1>   --wl <um>\n"
    "\n"
    "Calibration / weights:\n"
    "  --calfile <csv>        per-antenna per-pol per-channel weights (phasecal)\n"
    "\n"
    "Beamforming control:\n"
    "  --no-delay <0|1>       1=disable geometric delay/fringe-stopping (for testing; shows fringes)\n"
    "  --antmask <1111|1,0,1,1> enable antennas (ant0 must be 1)\n"
    "\n"
    "Monitoring / visdump:\n"
    "  --monitor <0|1>        print XMON lines (default 1)\n"
    "  --monitor-sec <sec>    monitor integration time (default 1)\n"
    "  --monitor-all <0|1>    print XMON for all dump targets (default 0)\n"
    "  --dump-baseline <i-j>  select baseline for XMON and --dump-vis (default 0-1)\n"
    "  --dump-pol <XX|YY|XY|YX>\n"
    "  --dump-vis <file>      write single visdump file\n"
    "  --dump-visdir <dir>    write ref->others visdump files into directory\n"
    "  --dump-ref <id>        reference antenna for --dump-visdir (default 0)\n"
    "  --dump-pols <list>     e.g. XX,YY (default XX,YY)\n"
    "\n"
    "CUDA:\n"
    "  --cpu-core <id>       bind process to CPU core id (Linux sched_setaffinity; default -1 disabled)\n"
    "  --cuda <id>            GPU id (required)\n"
    "  --cuda-list            list GPUs and exit\n"
    "  --batch-frames <N>     frames per batch (default 256)\n"
    "\n",
    prog);
}

// ------------------- CUDA kernels -------------------

__device__ __forceinline__ int16_t clip_i16(float x){
  // Round-to-nearest and saturate to int16.
  int v = __float2int_rn(x);
  if(v > 32767) v = 32767;
  if(v < -32768) v = -32768;
  return (int16_t)v;
}

// Unpack int16 complex samples into cufftComplex FFT input, applying coarse integer-sample delay.
// Layout per frame (PAYLOAD_SIZE=8192 bytes):
//   Xre[1024], Xim[1024], Yre[1024], Yim[1024] as int16.
__global__ void unpack_to_fft_input(const int16_t* __restrict__ in_raw,
                                    cufftComplex* __restrict__ fft,
                                    int frames,
                                    const int* __restrict__ coarse,
                                    int no_delay,
                                    int nfft,
                                    int nspec)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)NANT * (int64_t)NPOL * (int64_t)frames * (int64_t)nspec * (int64_t)nfft;
  if(idx >= total) return;

  int k = (int)(idx % nfft);
  int64_t b = idx / nfft;

  int sp = (int)(b % nspec);
  int64_t t = b / nspec;

  int f = (int)(t % frames);
  t /= frames;

  int p = (int)(t % NPOL);
  int a = (int)(t / NPOL);

  int n_out = sp * nfft + k;   // 0..FRAME_SAMP-1

  int n_src = n_out;
  if(!no_delay){
    n_src = n_out + coarse[a];
  }

  float re = 0.0f, im = 0.0f;

  if((unsigned)n_src < (unsigned)FRAME_SAMP){
    const int16_t* base = (const int16_t*)((const char*)in_raw + (size_t)a * (size_t)frames * (size_t)PAYLOAD_SIZE + (size_t)f * (size_t)SHORTS_PER_FRAME * sizeof(int16_t));
    const int16_t* Xre = base + 0*BLOCK_LEN;
    const int16_t* Xim = base + 1*BLOCK_LEN;
    const int16_t* Yre = base + 2*BLOCK_LEN;
    const int16_t* Yim = base + 3*BLOCK_LEN;
    if(p==0){ re = (float)Xre[n_src]; im = (float)Xim[n_src]; }
    else    { re = (float)Yre[n_src]; im = (float)Yim[n_src]; }
  }

  fft[idx].x = re;
  fft[idx].y = im;
}

__global__ void apply_weights(cufftComplex* __restrict__ fft,
                              const float2* __restrict__ ph,
                              const float2* __restrict__ cal,
                              int frames,
                              int use_geo,
                              int use_cal,
                              int nfft,
                              int nspec)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)NANT * (int64_t)NPOL * (int64_t)frames * (int64_t)nspec * (int64_t)nfft;
  if(idx >= total) return;

  int k = (int)(idx % nfft);
  int64_t b = idx / nfft;

  // sp is not used directly but kept for consistent indexing
  int sp = (int)(b % nspec);
  (void)sp;
  int64_t t = b / nspec;

  int f = (int)(t % frames);
  (void)f;
  t /= frames;

  int p = (int)(t % NPOL);
  int a = (int)(t / NPOL);

  float2 w; w.x = 1.0f; w.y = 0.0f;

  if(use_geo){
    float2 wg = ph[a * nfft + k];
    float wr = w.x*wg.x - w.y*wg.y;
    float wi = w.x*wg.y + w.y*wg.x;
    w.x = wr; w.y = wi;
  }
  if(use_cal){
    float2 wc = cal[((a * NPOL) + p) * nfft + k];
    float wr = w.x*wc.x - w.y*wc.y;
    float wi = w.x*wc.y + w.y*wc.x;
    w.x = wr; w.y = wi;
  }

  float xr = fft[idx].x;
  float xi = fft[idx].y;
  fft[idx].x = xr*w.x - xi*w.y;
  fft[idx].y = xr*w.y + xi*w.x;
}

__global__ void beamform_sum(const cufftComplex* __restrict__ fft,
                             cufftComplex* __restrict__ beam,
                             int frames,
                             float inv_nant_used,
                             int nfft,
                             int nspec)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)NPOL * (int64_t)frames * (int64_t)nspec * (int64_t)nfft;
  if(idx >= total) return;

  int k = (int)(idx % nfft);
  int64_t b2 = idx / nfft;

  int sp = (int)(b2 % nspec);
  int64_t t = b2 / nspec;

  int f = (int)(t % frames);
  int p = (int)(t / frames);

  float sum_re = 0.0f, sum_im = 0.0f;

  // sum antennas (inactive antennas are zero-filled on host)
  for(int a=0;a<NANT;a++){
    int64_t b = (((int64_t)a * NPOL + p) * frames + f) * nspec + sp;
    int64_t in_idx = b * nfft + k;
    sum_re += fft[in_idx].x;
    sum_im += fft[in_idx].y;
  }

  sum_re *= inv_nant_used;
  sum_im *= inv_nant_used;

  beam[idx].x = sum_re;
  beam[idx].y = sum_im;
}

__global__ void pack_output(const cufftComplex* __restrict__ beam_time,
                            int16_t* __restrict__ out_raw,
                            int frames,
                            float inv_nfft,
                            int nfft,
                            int nspec)
{
  // threads over frames*FRAME_SAMP
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)frames * (int64_t)FRAME_SAMP;
  if(idx >= total) return;

  int f = (int)(idx / FRAME_SAMP);
  int n = (int)(idx - (int64_t)f * FRAME_SAMP);

  int sp = n / nfft;
  int k  = n % nfft;

  // pol0 (X)
  int64_t b2x = ((int64_t)0 * frames + f) * nspec + sp;
  int64_t ix  = b2x * nfft + k;
  float xr = beam_time[ix].x * inv_nfft;
  float xi = beam_time[ix].y * inv_nfft;

  // pol1 (Y)
  int64_t b2y = ((int64_t)1 * frames + f) * nspec + sp;
  int64_t iy  = b2y * nfft + k;
  float yr = beam_time[iy].x * inv_nfft;
  float yi = beam_time[iy].y * inv_nfft;

  int16_t* base = out_raw + (int64_t)f * SHORTS_PER_FRAME;
  int16_t* Xre = base + 0*BLOCK_LEN;
  int16_t* Xim = base + 1*BLOCK_LEN;
  int16_t* Yre = base + 2*BLOCK_LEN;
  int16_t* Yim = base + 3*BLOCK_LEN;

  Xre[n] = clip_i16(xr);
  Xim[n] = clip_i16(xi);
  Yre[n] = clip_i16(yr);
  Yim[n] = clip_i16(yi);
}

__global__ void compute_vis_pow(const cufftComplex* __restrict__ fft,
                                float2* __restrict__ vis_out, // [nfft]
                                double* __restrict__ p0_out,  // [nfft]
                                double* __restrict__ p1_out,  // [nfft]
                                int frames,
                                int bi, int bj,
                                int pA, int pB,
                                int nfft,
                                int nspec)
{
  int k = threadIdx.x;
  if(k >= nfft) return;

  float v_re = 0.0f, v_im = 0.0f;
  double p0 = 0.0, p1 = 0.0;

  for(int f=0; f<frames; f++){
    for(int sp=0; sp<nspec; sp++){
      int64_t bA = (((int64_t)bi * NPOL + pA) * frames + f) * nspec + sp;
      int64_t bB = (((int64_t)bj * NPOL + pB) * frames + f) * nspec + sp;
      int64_t iA = bA * nfft + k;
      int64_t iB = bB * nfft + k;
      float ar = fft[iA].x, ai = fft[iA].y;
      float br = fft[iB].x, bi_ = fft[iB].y;

      // a * conj(b)
      v_re += ar*br + ai*bi_;
      v_im += ai*br - ar*bi_;

      p0 += (double)ar*(double)ar + (double)ai*(double)ai;
      p1 += (double)br*(double)br + (double)bi_*(double)bi_;
    }
  }

  vis_out[k].x = v_re;
  vis_out[k].y = v_im;
  p0_out[k] = p0;
  p1_out[k] = p1;
}


// ------------------- main -------------------
int main(int argc, char** argv)
{
  // Defaults: use spaced keys to avoid shm key collision on some psrdada builds.
  const char* in_keys[NANT] = {"a000","a002","a004","a006"};
  const char* out_key = "b000";

  double fs_hz = 32e6;
  double fcenter_hz = 1420e6;   // RF/sky center frequency (Hz)
  int nfft = NFFT_DEFAULT;      // FFT length (32/64/128/256/512/1024)
  double az_deg = 0.0;
  double el_deg = 90.0;
  int no_delay = 0;

  // Optional: track a sky position by ICRS RA/Dec (SOFA) instead of fixed Az/El
  const char* ra_s = nullptr;
  const char* dec_s = nullptr;
  double ra_rad = 0.0, dec_rad = 0.0;
  int use_radec = 0;
  double track_sec = 1.0;
  const char* track_csv_path = nullptr;  // optional: write per-update tracking CSV

  // SOFA EOP / atmosphere (defaults: 0 -> no correction/refraction)
  double dut1 = 0.0;        // UT1-UTC seconds
  double xp_arcsec = 0.0;   // polar motion x (arcsec)
  double yp_arcsec = 0.0;   // polar motion y (arcsec)
  double phpa = 0.0;        // pressure hPa (0 => no refraction)
  double tc = 0.0;          // temperature C
  double rh = 0.0;          // relative humidity 0..1
  double wl = 0.0;          // wavelength um

  int monitor = 1;
  double monitor_sec = 1.0;

  // SNR estimate mode for tracking CSV
  //   rad: use nominal B*T (radiometer) from fs and monitor-sec
  //   emp: use |V|/sigma with effective integration time inferred from processed frames
  int snr_mode = 1; // 0=rad, 1=emp (default)

  const char* dump_vis_path = nullptr;
  const char* antfile_path = nullptr;
  int ref_ant = 0;

  int dump_bi = 0, dump_bj = 1;
  int dump_pol_code = 0; // XX

  const char* calfile_path = nullptr;

  // Multi visdump (optional): write ref->other baselines to a directory
  const char* dump_vis_dir = nullptr;
  const char* dump_pols_str = "XX,YY";
  int dump_ref = 0;

  int monitor_all = 0;

  int cuda_dev = -1;
  int cuda_list = 0;

  int cpu_core = -1; // -1 disables CPU pinning

  int batch_frames = 256;

  int antmask[NANT] = {1,1,1,1};
  char antmask_str[32];
  antmask_to_str(antmask, antmask_str);

  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--in0") && i+1<argc) in_keys[0]=argv[++i];
    else if(!strcmp(argv[i],"--in1") && i+1<argc) in_keys[1]=argv[++i];
    else if(!strcmp(argv[i],"--in2") && i+1<argc) in_keys[2]=argv[++i];
    else if(!strcmp(argv[i],"--in3") && i+1<argc) in_keys[3]=argv[++i];
    else if(!strcmp(argv[i],"--out") && i+1<argc) out_key=argv[++i];
    else if(!strcmp(argv[i],"--fs") && i+1<argc) fs_hz = atof(argv[++i]);
    else if(!strcmp(argv[i],"--fcenter-hz") && i+1<argc) fcenter_hz = atof(argv[++i]);
    else if(!strcmp(argv[i],"--nfft") && i+1<argc) nfft = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--az") && i+1<argc) az_deg = atof(argv[++i]);
    else if(!strcmp(argv[i],"--el") && i+1<argc) el_deg = atof(argv[++i]);
    else if(!strcmp(argv[i],"--ra") && i+1<argc) ra_s = argv[++i];
    else if(!strcmp(argv[i],"--dec") && i+1<argc) dec_s = argv[++i];
    else if(!strcmp(argv[i],"--track-sec") && i+1<argc) track_sec = atof(argv[++i]);
    else if(!strcmp(argv[i],"--track-csv") && i+1<argc) track_csv_path = argv[++i];
    else if(!strcmp(argv[i],"--snr-mode") && i+1<argc){
      const char* m = argv[++i];
      std::string ms(m);
      for(auto &c: ms) c = (char)tolower((unsigned char)c);
      if(ms=="rad" || ms=="radiometer") snr_mode = 0;
      else if(ms=="emp" || ms=="vsigma" || ms=="vis") snr_mode = 1;
      else { fprintf(stderr,"[ERR] bad --snr-mode (use rad|emp)\n"); return 1; }
    }
    else if(!strcmp(argv[i],"--dut1") && i+1<argc) dut1 = atof(argv[++i]);
    else if(!strcmp(argv[i],"--xp") && i+1<argc) xp_arcsec = atof(argv[++i]);
    else if(!strcmp(argv[i],"--yp") && i+1<argc) yp_arcsec = atof(argv[++i]);
    else if(!strcmp(argv[i],"--phpa") && i+1<argc) phpa = atof(argv[++i]);
    else if(!strcmp(argv[i],"--tc") && i+1<argc) tc = atof(argv[++i]);
    else if(!strcmp(argv[i],"--rh") && i+1<argc) rh = atof(argv[++i]);
    else if(!strcmp(argv[i],"--wl") && i+1<argc) wl = atof(argv[++i]);
    else if(!strcmp(argv[i],"--no-delay") && i+1<argc) no_delay = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--monitor") && i+1<argc) monitor = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--monitor-sec") && i+1<argc) monitor_sec = atof(argv[++i]);
    else if(!strcmp(argv[i],"--dump-vis") && i+1<argc) dump_vis_path = argv[++i];
    else if(!strcmp(argv[i],"--dump-baseline") && i+1<argc){
      int bi,bj;
      if(!parse_baseline(argv[++i], bi, bj)){
        fprintf(stderr,"[ERR] bad --dump-baseline\n"); return 1;
      }
      dump_bi = bi; dump_bj = bj;
    }
    else if(!strcmp(argv[i],"--dump-pol") && i+1<argc){
      int pc = pol_code_from_str(argv[++i]);
      if(pc < 0){ fprintf(stderr,"[ERR] bad --dump-pol (use XX/YY/XY/YX)\n"); return 1; }
      dump_pol_code = pc;
    }
    else if(!strcmp(argv[i],"--calfile") && i+1<argc) calfile_path = argv[++i];
    else if(!strcmp(argv[i],"--dump-visdir") && i+1<argc) dump_vis_dir = argv[++i];
    else if(!strcmp(argv[i],"--dump-ref") && i+1<argc) dump_ref = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dump-pols") && i+1<argc) dump_pols_str = argv[++i];
    else if(!strcmp(argv[i],"--monitor-all") && i+1<argc) monitor_all = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--cuda") && i+1<argc) cuda_dev = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--cuda-list")) cuda_list = 1;
    else if(!strcmp(argv[i],"--cpu-core") && i+1<argc) cpu_core = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--batch-frames") && i+1<argc) batch_frames = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--antmask") && i+1<argc){
      if(!parse_antmask(argv[++i], antmask)){
        fprintf(stderr,"[ERR] bad --antmask (use 1111 or 1,0,1,1)\n");
        return 1;
      }
      antmask_to_str(antmask, antmask_str);
    }
    else if(!strcmp(argv[i],"--antfile") && i+1<argc) antfile_path = argv[++i];
    else if(!strcmp(argv[i],"--ref-ant") && i+1<argc) ref_ant = atoi(argv[++i]);
    else if(!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")){ usage(argv[0]); return 0; }
    else {
      fprintf(stderr,"[ERR] unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  // Optional: bind to a single CPU core (Linux).
  if(cpu_core >= 0){
    bind_cpu_core(cpu_core);
  }

  // RA/Dec tracking option
  if(ra_s || dec_s){
    if(!ra_s || !dec_s){
      fprintf(stderr,"[ERR] --ra and --dec must be provided together\n");
      return 1;
    }
    if(!parse_ra_hms_sofa(ra_s, &ra_rad)){
      fprintf(stderr,"[ERR] bad --ra format: %s (expect HH:MM:SS.S)\n", ra_s);
      return 1;
    }
    if(!parse_dec_dms_sofa(dec_s, &dec_rad)){
      fprintf(stderr,"[ERR] bad --dec format: %s (expect [+/-]DD:MM:SS.S)\n", dec_s);
      return 1;
    }
    use_radec = 1;
    if(track_sec <= 0) track_sec = 1.0;
  }

  if(cuda_list){
    int n=0;
    CUDA_OK(cudaGetDeviceCount(&n));
    fprintf(stderr,"CUDA devices: %d\n", n);
    for(int d=0; d<n; d++){
      cudaDeviceProp p;
      CUDA_OK(cudaGetDeviceProperties(&p, d));
      fprintf(stderr,"  [%d] %s  sm=%d.%d  mem=%.1f GiB\n",
              d, p.name, p.major, p.minor, (double)p.totalGlobalMem/(1024.0*1024.0*1024.0));
    }
    return 0;
  }

  // Validate nfft and derive nspec
  if(!(nfft==32 || nfft==64 || nfft==128 || nfft==256 || nfft==512 || nfft==1024)){
    fprintf(stderr,"[bf-cuda %s][ERR] unsupported --nfft=%d (use 32/64/128/256/512/1024)\n", BF_CUDA_VERSION, nfft);
    return 1;
  }
  if(nfft <= 0 || nfft > NFFT_MAX || (FRAME_SAMP % nfft) != 0){
    fprintf(stderr,"[bf-cuda %s][ERR] invalid --nfft=%d (FRAME_SAMP=%d must be divisible)\n", BF_CUDA_VERSION, nfft, FRAME_SAMP);
    return 1;
  }
  int nspec = FRAME_SAMP / nfft;
  if(fcenter_hz <= 0.0){
    fprintf(stderr,"[bf-cuda %s][ERR] --fcenter-hz must be > 0\n", BF_CUDA_VERSION);
    return 1;
  }

  // Optional: load antenna positions from geodetic CSV (--antfile) and compute ENU
  if(antfile_path){
    if(!load_antfile_geo_csv(antfile_path, ref_ant)){
      fprintf(stderr, "[bf-cuda %s][ERR] failed to load antenna file\n", BF_CUDA_VERSION);
      return 1;
    }
  } else {
    fprintf(stderr, "[bf-cuda %s] Using built-in ENU antenna positions (no --antfile)\n", BF_CUDA_VERSION);
    for(int a=0;a<NANT;a++){
      fprintf(stderr, "[bf-cuda %s] ant%d ENU E=%.3f N=%.3f U=%.3f m\n",
              BF_CUDA_VERSION, a, ant_enu[a][0], ant_enu[a][1], ant_enu[a][2]);
    }
  }


  // Antenna enable mask
  if(antmask[0] != 1){
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][ERR] ant0 is the reference antenna and must be enabled (mask[0]=1)\n");
    return 1;
  }
  std::vector<int> active_ants;
  for(int a=0;a<NANT;a++) if(antmask[a]) active_ants.push_back(a);
  int nant_used = (int)active_ants.size();
  if(nant_used <= 0){
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][ERR] no active antennas in --antmask\n");
    return 1;
  }
  float inv_nant_used = 1.0f / (float)nant_used;
  fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] antmask=%s  nant_used=%d\n", antmask_str, nant_used);


  if(cuda_dev < 0){
    fprintf(stderr,"[ERR] --cuda <id> is required. Use --cuda-list to view GPUs.\n");
    return 1;
  }

  if(batch_frames <= 0) batch_frames = 256;
  if((batch_frames % 1) != 0) batch_frames = 256;

  // device select
  int ndev=0;
  CUDA_OK(cudaGetDeviceCount(&ndev));
  if(cuda_dev >= ndev){
    fprintf(stderr,"[ERR] --cuda %d out of range (device_count=%d)\n", cuda_dev, ndev);
    return 1;
  }
  CUDA_OK(cudaSetDevice(cuda_dev));
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, cuda_dev));
  fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] using GPU %d: %s\n", cuda_dev, prop.name);

  if(monitor_sec <= 0) monitor_sec = 1.0;

  // delay model (initial). If --ra/--dec are provided and --no-delay=0,
  // we compute current Az/El via SOFA (system UTC) here. Inside each RUN loop
  // we will re-sync weights to UTC_START (preferred) and keep updating every --track-sec.
  double tau[NANT]={0};
  int coarse[NANT]={0};
  double fine[NANT]={0};

  // polar motion arcsec -> rad
  double xp_rad = xp_arcsec * (deg2rad(1.0) / 3600.0);
  double yp_rad = yp_arcsec * (deg2rad(1.0) / 3600.0);

  if(use_radec){
    if(!ant_geo_valid){
      fprintf(stderr,"[bf-cuda %s][ERR] --ra/--dec requires --antfile with geodetic coordinates\n", BF_CUDA_VERSION);
      return 1;
    }
    int y,mo,day,hh,mm; double sec;
    get_system_utc_ymdhms(&y,&mo,&day,&hh,&mm,&sec);
    double utc1=0.0, utc2=0.0;
    if(!sofa_utc_to_jd(y,mo,day,hh,mm,sec,&utc1,&utc2)){
      fprintf(stderr,"[bf-cuda %s][ERR] iauDtf2d(UTC) failed for system time\n", BF_CUDA_VERSION);
      return 1;
    }
    double az_all[NANT]={0}, el_all[NANT]={0};
    if(!sofa_radec_to_azel_all(ra_rad, dec_rad, utc1, utc2, dut1,
                               xp_rad, yp_rad, phpa, tc, rh, wl,
                               az_all, el_all)){
      fprintf(stderr,"[bf-cuda %s][ERR] SOFA radec->azel failed\n", BF_CUDA_VERSION);
      return 1;
    }
    az_deg = az_all[0];
    el_deg = el_all[0];
    fprintf(stderr,"[bf-cuda %s] pointing from RA/Dec (system UTC): RA=%s Dec=%s => ant0 az=%.3f el=%.3f\n",
            BF_CUDA_VERSION, ra_s, dec_s, az_deg, el_deg);
  }

  geometric_delays(az_deg, el_deg, tau);
  for(int a=0;a<NANT;a++){
    double x = tau[a]*fs_hz;
    coarse[a] = (int)llround(x);
    fine[a] = tau[a] - ((double)coarse[a]/fs_hz);
  }

  if(use_radec){
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] fs=%.3f MHz fcenter=%.3f MHz nfft=%d df=%.3f kHz batch_frames=%d no_delay=%d track RA=%s Dec=%s track_sec=%.3f (ant0 az=%.3f el=%.3f)\n",
            fs_hz/1e6, fcenter_hz/1e6, nfft, (fs_hz/(double)nfft)/1e3, batch_frames, no_delay, ra_s, dec_s, track_sec, az_deg, el_deg);
  } else {
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] fs=%.3f MHz fcenter=%.3f MHz nfft=%d df=%.3f kHz batch_frames=%d no_delay=%d steer az=%.3f el=%.3f\n",
            fs_hz/1e6, fcenter_hz/1e6, nfft, (fs_hz/(double)nfft)/1e3, batch_frames, no_delay, az_deg, el_deg);
  }

  // signed FFT bin frequencies (Hz) for channel index k (FFT output order, not fftshifted)
  std::vector<double> fbin(nfft);
  for(int k=0;k<nfft;k++){
    int kk = (k <= (nfft/2)) ? k : (k - nfft);
    fbin[k] = (double)kk * (fs_hz / (double)nfft);
  }

  // Geometric phasor table per-antenna per-channel.
  // We apply:
  //   - coarse integer-sample shift in time domain
  //   - fractional delay via exp(-j 2π f_bin * fine)
  //   - carrier/fringe phase via exp(-j 2π f_center * tau)
  std::vector<float2> h_ph((size_t)NANT * (size_t)nfft);
  for(int a=0;a<NANT;a++){
    double phase0 = -2.0 * M_PI * fcenter_hz * tau[a];
    for(int k=0;k<nfft;k++){
      double phase = phase0 - 2.0 * M_PI * fbin[k] * fine[a];
      float2 w; w.x = (float)cos(phase); w.y = (float)sin(phase);
      h_ph[(size_t)a * (size_t)nfft + (size_t)k] = w;
    }
  }

  // Visibility / VISDUMP targets
  // Selected baseline/pol for XMON printing (and for --dump-vis single output)
  if(dump_bi < 0 || dump_bi >= NANT || dump_bj < 0 || dump_bj >= NANT){
    fprintf(stderr,"[ERR] --dump-baseline indices must be within 0..%d\n", NANT-1);
    return 1;
  }

  bool want_any_vis = (monitor || dump_vis_path || dump_vis_dir || track_csv_path);
  if(want_any_vis && (nant_used < 2)){
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] only one active antenna; disabling XMON/VISDUMP.\n");
    monitor = 0;
    dump_vis_path = nullptr;
    dump_vis_dir = nullptr;
    want_any_vis = false;
  }

  // If requested monitor/single-dump baseline uses a disabled antenna, auto-select first two active antennas.
  if(want_any_vis && (monitor || dump_vis_path) && (!antmask[dump_bi] || !antmask[dump_bj])){
    int old_bi = dump_bi, old_bj = dump_bj;
    dump_bi = active_ants[0];
    dump_bj = active_ants[1];
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] baseline %d-%d uses disabled antenna(s). Using %d-%d for XMON/--dump-vis.\n",
            old_bi, old_bj, dump_bi, dump_bj);
  }

  // Build a unified list of vis targets (for monitor and for file dumps)
  std::vector<dump_target_t> dump_targets;
  auto add_target = [&](int bi, int bj, int pol_code, FILE* fp){
    for(auto &t : dump_targets){
      if(t.bi==bi && t.bj==bj && t.pol_code==pol_code){
        if(!fp){
          // monitor-only request; existing target is fine
          return;
        }
        if(!t.fp){
          // attach this file handle to existing target
          t.fp = fp;
          return;
        }
        // existing already has a file handle -> keep this as an additional target (do not merge)
      }
    }
    dump_target_t t;
    t.bi = bi; t.bj = bj; t.pol_code = pol_code;
    polcode_to_pab(pol_code, t.pA, t.pB);
    t.fp = fp;
    t.vis_acc.assign(nfft, cf32{0.0f,0.0f});
    t.p0_acc.assign(nfft, 0.0);
    t.p1_acc.assign(nfft, 0.0);
    t.frames_acc = 0;
    t.have_last = 0;
    t.last_coh_avg = (float)NAN;
    t.last_snr_rad = (float)NAN;
    t.last_snr_emp = (float)NAN;
    t.last_bw_eff_hz = NAN;
    t.last_tint_emp = NAN;
    t.last_tau_res = NAN;
    t.last_p0_mean = NAN;
    t.last_p1_mean = NAN;
    t.last_t_mon = NAN;
    dump_targets.push_back(std::move(t));
  };

  // Single visdump file (backward compatible)
  if(dump_vis_path){
    FILE* fp = fopen(dump_vis_path, "wb");
    if(!fp){
      fprintf(stderr,"[ERR] cannot open --dump-vis %s: %s\n", dump_vis_path, strerror(errno));
      return 1;
    }
    write_visdump_header(fp, fs_hz, monitor_sec, (uint32_t)dump_bi, (uint32_t)dump_bj, (uint32_t)dump_pol_code, (uint32_t)nfft);
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] dumping vis to %s (baseline=%d-%d pol=%s)\n",
            dump_vis_path, dump_bi, dump_bj, polcode_to_str(dump_pol_code));
    add_target(dump_bi, dump_bj, dump_pol_code, fp);
  }

  // Multi visdump directory: dump (dump_ref -> each other enabled ant) for selected pol products
  if(dump_vis_dir){
    if(dump_ref < 0 || dump_ref >= NANT || !antmask[dump_ref]){
      fprintf(stderr,"[ERR] --dump-ref %d invalid or disabled by --antmask\n", dump_ref);
      return 1;
    }
    if(!ensure_dir_exists(dump_vis_dir)){
      return 1;
    }
    std::vector<int> pol_list;
    if(!parse_pol_list(dump_pols_str, pol_list)){
      fprintf(stderr,"[ERR] bad --dump-pols '%s' (example: XX,YY)\n", dump_pols_str ? dump_pols_str : "");
      return 1;
    }
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] dump-visdir=%s ref=%d pols=%s\n", dump_vis_dir, dump_ref, dump_pols_str);
    for(int a=0;a<NANT;a++){
      if(a == dump_ref) continue;
      if(!antmask[a]) continue;
      for(int pc : pol_list){
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/vis_%d-%d_%s.bin", dump_vis_dir, dump_ref, a, polcode_to_str(pc));
        FILE* fp = fopen(fname, "wb");
        if(!fp){
          fprintf(stderr,"[ERR] cannot open %s: %s\n", fname, strerror(errno));
          return 1;
        }
        write_visdump_header(fp, fs_hz, monitor_sec, (uint32_t)dump_ref, (uint32_t)a, (uint32_t)pc, (uint32_t)nfft);
        fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] dumping vis to %s (baseline=%d-%d pol=%s)\n",
                fname, dump_ref, a, polcode_to_str(pc));
        add_target(dump_ref, a, pc, fp);
      }
    }
  }

  // Ensure monitor baseline exists in dump_targets (even without file dumps)
  if(monitor){
    add_target(dump_bi, dump_bj, dump_pol_code, nullptr);
  }
  // If tracking CSV is enabled, ensure ref->others baselines exist as monitor targets
  // so coh/power columns can be filled even when not dumping VIS files.
  if(track_csv_path && track_csv_path[0]){
    int ref = 0;  // tracking CSV uses ant0 as reference
    if(ref < 0 || ref >= NANT || !antmask[ref]){
      ref = active_ants[0];
      fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] --dump-ref invalid/disabled; using ref=%d for --track-csv targets\n", ref);
    }
    std::vector<int> pol_list;
    if(!parse_pol_list(dump_pols_str, pol_list)){
      fprintf(stderr,"[ERR] bad --dump-pols '%s' (example: XX,YY)\n", dump_pols_str ? dump_pols_str : "");
      return 1;
    }
    for(int a=0;a<NANT;a++){
      if(a==ref) continue;
      if(!antmask[a]) continue;
      for(int pc : pol_list){
        add_target(ref, a, pc, nullptr);
      }
    }
  }



  // Load calibration weights (phasecal) if provided
  std::vector<float2> h_cal;
  if(!load_calfile_csv(calfile_path, nfft, fs_hz, h_cal)){
    return 1;
  }
  if(!calfile_path){
    fprintf(stderr,"[bf-cuda %s] no --calfile (unity weights)\n", BF_CUDA_VERSION);
  }



  // psrdada connect
  multilog_t* log = multilog_open("bf_fxbv_stream_dumpvis_cuda", 0);
  multilog_add(log, stderr);

  dada_hdu_t* in_hdu[NANT] = {nullptr,nullptr,nullptr,nullptr};
  for(int a=0;a<NANT;a++){
    if(!antmask[a]) continue;
    in_hdu[a] = dada_hdu_create(log);
    dada_hdu_set_key(in_hdu[a], parse_key(in_keys[a]));
    if(dada_hdu_connect(in_hdu[a]) < 0){
      multilog(log, LOG_ERR, "connect input %s failed (ring not created?)\n", in_keys[a]);
      return 1;
    }
  }

  dada_hdu_t* out_hdu = dada_hdu_create(log);
  dada_hdu_set_key(out_hdu, parse_key(out_key));
  if(dada_hdu_connect(out_hdu) < 0){
    multilog(log, LOG_ERR, "connect output %s failed (ring not created?)\n", out_key);
    return 1;
  }

  // host pinned buffers
  int16_t* h_in = nullptr;
  int16_t* h_out = nullptr;

  size_t in_bytes = (size_t)NANT * (size_t)batch_frames * PAYLOAD_SIZE;
  size_t out_bytes = (size_t)batch_frames * PAYLOAD_SIZE;

  CUDA_OK(cudaMallocHost((void**)&h_in, in_bytes));
  CUDA_OK(cudaMallocHost((void**)&h_out, out_bytes));

  // device buffers
  int16_t* d_in = nullptr;
  int16_t* d_out = nullptr;
  cufftComplex* d_fft = nullptr;
  cufftComplex* d_beam = nullptr;
  float2* d_ph = nullptr;
  float2* d_cal = nullptr;
  int* d_coarse = nullptr;

  float2* d_vis = nullptr;
  double* d_p0 = nullptr;
  double* d_p1 = nullptr;

  CUDA_OK(cudaMalloc((void**)&d_in, in_bytes));
  CUDA_OK(cudaMalloc((void**)&d_out, out_bytes));

  int64_t batch_total = (int64_t)NANT * NPOL * batch_frames * nspec;
  int64_t batch2_total = (int64_t)NPOL * batch_frames * nspec;

  CUDA_OK(cudaMalloc((void**)&d_fft, (size_t)batch_total * nfft * sizeof(cufftComplex)));
  CUDA_OK(cudaMalloc((void**)&d_beam, (size_t)batch2_total * nfft * sizeof(cufftComplex)));

  CUDA_OK(cudaMalloc((void**)&d_ph, NANT * nfft * sizeof(float2)));
  CUDA_OK(cudaMemcpy(d_ph, h_ph.data(), NANT * nfft * sizeof(float2), cudaMemcpyHostToDevice));

  if(calfile_path){
    CUDA_OK(cudaMalloc((void**)&d_cal, (size_t)NANT * NPOL * nfft * sizeof(float2)));
    CUDA_OK(cudaMemcpy(d_cal, h_cal.data(), (size_t)NANT * NPOL * nfft * sizeof(float2), cudaMemcpyHostToDevice));
  }

  CUDA_OK(cudaMalloc((void**)&d_coarse, NANT * sizeof(int)));
  CUDA_OK(cudaMemcpy(d_coarse, coarse, NANT * sizeof(int), cudaMemcpyHostToDevice));

  CUDA_OK(cudaMalloc((void**)&d_vis, nfft * sizeof(float2)));
  CUDA_OK(cudaMalloc((void**)&d_p0, nfft * sizeof(double)));
  CUDA_OK(cudaMalloc((void**)&d_p1, nfft * sizeof(double)));

  // host arrays for VIS/XMON per target
  size_t ntarget = dump_targets.size();
  std::vector<cf32> vis_frame(ntarget * nfft);
  std::vector<double> p0_frame(ntarget * nfft);
  std::vector<double> p1_frame(ntarget * nfft);

  // cuFFT plans
  cufftHandle plan_fwd, plan_inv;
  CUFFT_OK(cufftPlan1d(&plan_fwd, nfft, CUFFT_C2C, (int)batch_total));
  CUFFT_OK(cufftPlan1d(&plan_inv, nfft, CUFFT_C2C, (int)batch2_total));

  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_OK(cufftSetStream(plan_fwd, stream));
  CUFFT_OK(cufftSetStream(plan_inv, stream));

  // signal
  signal(SIGINT, on_sigint);
  signal(SIGTERM, on_sigint);

  double t0 = now_sec();
  double t_last_mon = t0;

  fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] entering RUN loop. waiting for input headers...\n");

  while(!g_stop){
    int in_used[NANT] = {0,0,0,0};
    int in_locked[NANT] = {0,0,0,0};
    int out_used = 0;
    int out_locked = 0;
    FILE* fp_track_csv = nullptr;

    // lock inputs
    for(int a=0;a<NANT;a++){
      if(!antmask[a]) continue;
      fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] lock_read %s ...\n", in_keys[a]);
      if(dada_hdu_lock_read(in_hdu[a]) < 0){
        multilog(log, LOG_ERR, "lock_read input %s failed (another reader attached?)\n", in_keys[a]);
        goto run_fail;
      }
      in_locked[a] = 1;
    }
    // lock output
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] lock_write %s ...\n", out_key);
    if(dada_hdu_lock_write(out_hdu) < 0){
      multilog(log, LOG_ERR, "lock_write output %s failed (another writer attached?)\n", out_key);
      goto run_fail;
    }
    out_locked = 1;

    // read headers
    char hdr_in0[4096];
    for(int a=0;a<NANT;a++){
      if(!antmask[a]) continue;
      fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] wait header %s ...\n", in_keys[a]);
      uint64_t hdr_sz = 0;
      char* hp = ipcbuf_get_next_read(in_hdu[a]->header_block, &hdr_sz);
      if(!hp){
        multilog(log, LOG_ERR, "get header from %s failed\n", in_keys[a]);
        goto run_fail;
      }
      if(a==0) memcpy(hdr_in0, hp, 4096);
      ipcbuf_mark_cleared(in_hdu[a]->header_block);
    }

    // write output header (copy in0 and annotate)
    char hdr_out[4096];
    memcpy(hdr_out, hdr_in0, 4096);
    ascii_header_set(hdr_out, "INSTRUMENT", "%s", "MedusaBF-CUDA");
    ascii_header_set(hdr_out, "BEAMFORMED", "%d", 1);
    ascii_header_set(hdr_out, "BEAM_ID", "%d", 0);
    ascii_header_set(hdr_out, "BF_NFFT", "%d", nfft);
    ascii_header_set(hdr_out, "BF_NSPEC", "%d", nspec);
    ascii_header_set(hdr_out, "BF_FS_HZ", "%.6f", fs_hz);
    ascii_header_set(hdr_out, "BF_FCENTER_HZ", "%.6f", fcenter_hz);
    ascii_header_set(hdr_out, "BF_DF_HZ", "%.6f", fs_hz/(double)nfft);
    ascii_header_set(hdr_out, "BF_AZ", "%.6f", az_deg);
    ascii_header_set(hdr_out, "BF_EL", "%.6f", el_deg);
    ascii_header_set(hdr_out, "BF_RA", "%s", ra_s ? ra_s : "none");
    ascii_header_set(hdr_out, "BF_DEC", "%s", dec_s ? dec_s : "none");
    ascii_header_set(hdr_out, "BF_TRACK_SEC", "%.3f", track_sec);
    ascii_header_set(hdr_out, "BF_TRACK_CSV", "%s", track_csv_path ? track_csv_path : "none");
    ascii_header_set(hdr_out, "BF_SNR_MODE", "%s", (snr_mode==0) ? "rad" : "emp");
    ascii_header_set(hdr_out, "BF_DUT1", "%.6f", dut1);
    ascii_header_set(hdr_out, "BF_XP_ARCSEC", "%.6f", xp_arcsec);
    ascii_header_set(hdr_out, "BF_YP_ARCSEC", "%.6f", yp_arcsec);
    ascii_header_set(hdr_out, "BF_NO_DELAY", "%d", no_delay);
    ascii_header_set(hdr_out, "BF_CUDA_DEV", "%d", cuda_dev);
    ascii_header_set(hdr_out, "BF_BATCH_FRAMES", "%d", batch_frames);
    ascii_header_set(hdr_out, "BF_ANTMASK", "%s", antmask_str);
    ascii_header_set(hdr_out, "BF_NANT_USED", "%d", nant_used);
    ascii_header_set(hdr_out, "BF_CALFILE", "%s", calfile_path ? calfile_path : "none");
    ascii_header_set(hdr_out, "BF_DUMP_VIS", "%s", dump_vis_path ? dump_vis_path : "none");
    ascii_header_set(hdr_out, "BF_DUMP_VISDIR", "%s", dump_vis_dir ? dump_vis_dir : "none");
    ascii_header_set(hdr_out, "BF_DUMP_REF", "%d", dump_ref);
    ascii_header_set(hdr_out, "BF_DUMP_POLS", "%s", dump_pols_str ? dump_pols_str : "");

    char* hwr;
    hwr = ipcbuf_get_next_write(out_hdu->header_block);
    if(!hwr){
      multilog(log, LOG_ERR, "get output header write failed\n");
      goto run_fail;
    }
    memcpy(hwr, hdr_out, 4096);
    ipcbuf_mark_filled(out_hdu->header_block, 4096);

    char utc_start[128];
    utc_start[0] = '\0';
    if(ascii_header_get(hdr_in0, "UTC_START", "%127s", utc_start))
      fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] RUN started. UTC_START=%s\n", utc_start);
    else
      fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] RUN started. UTC_START=(unknown)\n");

    // RA/Dec tracking base time (Julian Date UTC). Prefer UTC_START from header.
    double utc1_base, utc2_base;
    int have_utc_base;
    utc1_base = 0.0; utc2_base = 0.0;
    have_utc_base = 0;
    if(use_radec){
      int y,mo,day,hh,mm; double sec;
      if(parse_utc_ymdhms(utc_start, &y,&mo,&day,&hh,&mm,&sec) && sofa_utc_to_jd(y,mo,day,hh,mm,sec,&utc1_base,&utc2_base)){
        have_utc_base = 1;
        fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] tracking time source: UTC_START (JD=%.6f + %.6f)\n", utc1_base, utc2_base);
      } else {
        have_utc_base = 0;
        fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] UTC_START parse failed; tracking uses system UTC\n");
      }
    }

    uint64_t frames_total;         // frames processed in this RUN
    double t_last_track_data;  // seconds (data time) of last tracking update
    frames_total = 0;
    t_last_track_data = -1e99;

    // open tracking CSV if requested (append; write header if new/empty)
    if(track_csv_path && track_csv_path[0]){
      struct stat st_csv;
      int need_hdr = (stat(track_csv_path, &st_csv) != 0) || (st_csv.st_size == 0);
      fp_track_csv = fopen(track_csv_path, "a");
      if(!fp_track_csv){
        fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] cannot open --track-csv %s: %s\n", track_csv_path, strerror(errno));
      } else {
        if(need_hdr){
          fprintf(fp_track_csv, "utc_iso,t_data_sec,ant_id,az_deg,el_deg,tau_ns,coarse_samp,fine_ns,coh_xx,snr_xx,snr_xx_rad,snr_xx_emp,p_ref_xx,p_ant_xx,tau_res_xx_ns,coh_yy,snr_yy,snr_yy_rad,snr_yy_emp,p_ref_yy,p_ant_yy,tau_res_yy_ns\n");
          fflush(fp_track_csv);
        }
        fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] tracking CSV -> %s\n", track_csv_path);
      }
    }

    // reset VIS/XMON accumulators at run start
    for(auto &t : dump_targets){
      std::fill(t.vis_acc.begin(), t.vis_acc.end(), cf32{0.0f,0.0f});
      std::fill(t.p0_acc.begin(), t.p0_acc.end(), 0.0);
      std::fill(t.p1_acc.begin(), t.p1_acc.end(), 0.0);
      t.frames_acc = 0;
    }

    // process batches until EOD
    while(!g_stop){
      // read batch from each antenna (inactive antennas are zero-filled)
      for(int a=0;a<NANT;a++){
        char* dst = (char*)h_in + (size_t)a * batch_frames * PAYLOAD_SIZE;
        if(!antmask[a]){
          memset(dst, 0, (size_t)batch_frames * PAYLOAD_SIZE);
          in_used[a] = 0;
          continue;
        }
        int rc = read_exact(in_hdu[a]->data_block, dst, (size_t)batch_frames * PAYLOAD_SIZE);
        if(rc < 0){
          multilog(log, LOG_ERR, "read error on %s (ipcio_read)\n", in_keys[a]);
          goto run_fail;
        }
        if(rc == 0){
          fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] EOD on %s -> end run\n", in_keys[a]);
          goto run_ok;
        }
        in_used[a] = 1;
      }

      // update geometric delay/fringe-stopping from RA/Dec (SOFA) on a data-time cadence
      if(use_radec){
        double t_mid_data = ((double)frames_total + 0.5*(double)batch_frames) * (double)FRAME_SAMP / fs_hz;
        if((t_mid_data - t_last_track_data) >= track_sec){
          double utc1=0.0, utc2=0.0;
          if(have_utc_base){
            utc1 = utc1_base;
            utc2 = utc2_base + t_mid_data/86400.0;
          } else {
            int y,mo,day,hh,mm; double sec;
            get_system_utc_ymdhms(&y,&mo,&day,&hh,&mm,&sec);
            if(!sofa_utc_to_jd(y,mo,day,hh,mm,sec,&utc1,&utc2)){
              fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] iauDtf2d failed for system UTC; skip tracking update\n");
              utc1 = 0.0; utc2 = 0.0;
            }
          }

          double az_all[NANT]={0}, el_all[NANT]={0};
          if((utc1!=0.0) || (utc2!=0.0)){
            if(sofa_radec_to_azel_all(ra_rad, dec_rad, utc1, utc2, dut1,
                                      xp_rad, yp_rad, phpa, tc, rh, wl,
                                      az_all, el_all)){
              az_deg = az_all[0];
              el_deg = el_all[0];
              geometric_delays(az_deg, el_deg, tau);
              for(int a=0;a<NANT;a++){
                double x = tau[a]*fs_hz;
                coarse[a] = (int)llround(x);
                fine[a] = tau[a] - ((double)coarse[a]/fs_hz);
              }
              for(int a=0;a<NANT;a++){
                double phase0 = -2.0*M_PI * fcenter_hz * tau[a];
                for(int k=0;k<nfft;k++){
                  double phase = phase0 - 2.0*M_PI * fbin[k] * fine[a];
                  h_ph[(size_t)a*(size_t)nfft + (size_t)k] = make_float2((float)cos(phase),(float)sin(phase));
                }
              }
              if(!no_delay){
              CUDA_OK(cudaMemcpyAsync(d_coarse, coarse, NANT*sizeof(int), cudaMemcpyHostToDevice, stream));
              CUDA_OK(cudaMemcpyAsync(d_ph, h_ph.data(), NANT*nfft*sizeof(float2), cudaMemcpyHostToDevice, stream));
              }

              if(monitor){
                if(have_utc_base)
                  fprintf(stderr,"[TRACK] t_data=%.3fs (UTC_START+%.3fs) ant0 az=%.3f el=%.3f\n", t_mid_data, t_mid_data, az_deg, el_deg);
                else
                  fprintf(stderr,"[TRACK] system-UTC ant0 az=%.3f el=%.3f (t_data=%.3fs)\n", az_deg, el_deg, t_mid_data);
                fprintf(stderr,"[TRACK] AzEl(deg): a0 %.3f %.3f | a1 %.3f %.3f | a2 %.3f %.3f | a3 %.3f %.3f\n",
                        az_all[0],el_all[0], az_all[1],el_all[1], az_all[2],el_all[2], az_all[3],el_all[3]);
                fprintf(stderr,"[TRACK] tau(ns): a0 %.3f a1 %.3f a2 %.3f a3 %.3f  coarse(samp): %d %d %d %d\n",
                        tau[0]*1e9, tau[1]*1e9, tau[2]*1e9, tau[3]*1e9,
                        coarse[0], coarse[1], coarse[2], coarse[3]);
              }

              // write tracking CSV rows (one per active antenna)
              if(fp_track_csv){
                char utc_iso[64];
                if(!sofa_jd_to_utc_iso(utc1, utc2, utc_iso, sizeof(utc_iso))){
                  snprintf(utc_iso, sizeof(utc_iso), "JD%.6f+%.6f", utc1, utc2);
                }
                int ref = 0;  // ant0 reference for tau/coarse/fine
                if(ref < 0 || ref >= NANT) ref = 0;
                for(int a=0;a<NANT;a++){
                  if(!antmask[a]) continue;
                  double coh_xx=NAN, p_ref_xx=NAN, p_ant_xx=NAN, tau_res_xx_ns=NAN;
                  double snr_xx=NAN, snr_xx_rad=NAN, snr_xx_emp=NAN;
                  double coh_yy=NAN, p_ref_yy=NAN, p_ant_yy=NAN, tau_res_yy_ns=NAN;
                  double snr_yy=NAN, snr_yy_rad=NAN, snr_yy_emp=NAN;
                  if(a != ref){
                    dump_target_t* txx = find_target(dump_targets, ref, a, 0); // XX
                    if(txx && txx->have_last){
                      coh_xx = (double)txx->last_coh_avg;
                      snr_xx_rad = (double)txx->last_snr_rad;
                      snr_xx_emp = (double)txx->last_snr_emp;
                      snr_xx = (snr_mode==0) ? snr_xx_rad : snr_xx_emp;
                      p_ref_xx = txx->last_p0_mean;
                      p_ant_xx = txx->last_p1_mean;
                      tau_res_xx_ns = txx->last_tau_res * 1e9;
                    }
                    dump_target_t* tyy = find_target(dump_targets, ref, a, 1); // YY
                    if(tyy && tyy->have_last){
                      coh_yy = (double)tyy->last_coh_avg;
                      snr_yy_rad = (double)tyy->last_snr_rad;
                      snr_yy_emp = (double)tyy->last_snr_emp;
                      snr_yy = (snr_mode==0) ? snr_yy_rad : snr_yy_emp;
                      p_ref_yy = tyy->last_p0_mean;
                      p_ant_yy = tyy->last_p1_mean;
                      tau_res_yy_ns = tyy->last_tau_res * 1e9;
                    }
                  }
                  fprintf(fp_track_csv,
                          "%s,%.6f,%d,%.6f,%.6f,%.6f,%d,%.6f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                          utc_iso, t_mid_data, a,
                          az_all[a], el_all[a],
                          tau[a]*1e9, coarse[a], fine[a]*1e9,
                          coh_xx, snr_xx, snr_xx_rad, snr_xx_emp, p_ref_xx, p_ant_xx, tau_res_xx_ns,
                          coh_yy, snr_yy, snr_yy_rad, snr_yy_emp, p_ref_yy, p_ant_yy, tau_res_yy_ns);
                }
                fflush(fp_track_csv);
              }
              t_last_track_data = t_mid_data;
            } else {
              fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][WARN] SOFA radec->azel failed; skip tracking update\n");
            }
          }
        }
      }


      // H->D
      CUDA_OK(cudaMemcpyAsync(d_in, h_in, in_bytes, cudaMemcpyHostToDevice, stream));

      // unpack + FFT forward
      int threads = 256;
      int64_t total_fft_elems = (int64_t)NANT * NPOL * batch_frames * nspec * nfft;
      int blocks_fft = (int)((total_fft_elems + threads - 1) / threads);
      unpack_to_fft_input<<<blocks_fft, threads, 0, stream>>>(d_in, d_fft, batch_frames, d_coarse, no_delay, nfft, nspec);
      CUDA_OK(cudaGetLastError());

      CUFFT_OK(cufftExecC2C(plan_fwd, d_fft, d_fft, CUFFT_FORWARD));
      int use_geo = (!no_delay);
      int use_cal = (d_cal != nullptr);
      if(use_geo || use_cal){
        apply_weights<<<blocks_fft, threads, 0, stream>>>(d_fft, d_ph, d_cal, batch_frames, use_geo, use_cal, nfft, nspec);
        CUDA_OK(cudaGetLastError());
      }

      // compute vis/pow for each target (optional; nfft=32)
      if(ntarget > 0){
        for(size_t ti=0; ti<ntarget; ti++){
          const auto &t = dump_targets[ti];
          compute_vis_pow<<<1, nfft, 0, stream>>>(d_fft, d_vis, d_p0, d_p1, batch_frames, t.bi, t.bj, t.pA, t.pB, nfft, nspec);
          CUDA_OK(cudaGetLastError());
          CUDA_OK(cudaMemcpyAsync((void*)(vis_frame.data() + ti*nfft), d_vis, nfft*sizeof(float2), cudaMemcpyDeviceToHost, stream));
          CUDA_OK(cudaMemcpyAsync((void*)(p0_frame.data() + ti*nfft), d_p0, nfft*sizeof(double), cudaMemcpyDeviceToHost, stream));
          CUDA_OK(cudaMemcpyAsync((void*)(p1_frame.data() + ti*nfft), d_p1, nfft*sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
      }

      // beamform sum
      int64_t total_beam_elems = (int64_t)NPOL * batch_frames * nspec * nfft;
      int blocks_beam = (int)((total_beam_elems + threads - 1) / threads);
      beamform_sum<<<blocks_beam, threads, 0, stream>>>(d_fft, d_beam, batch_frames, inv_nant_used, nfft, nspec);
      CUDA_OK(cudaGetLastError());

      // IFFT
      CUFFT_OK(cufftExecC2C(plan_inv, d_beam, d_beam, CUFFT_INVERSE));

      // pack output
      int64_t total_samples = (int64_t)batch_frames * FRAME_SAMP;
      int blocks_pack = (int)((total_samples + threads - 1) / threads);
      pack_output<<<blocks_pack, threads, 0, stream>>>(d_beam, d_out, batch_frames, 1.0f/(float)nfft, nfft, nspec);
      CUDA_OK(cudaGetLastError());      // D->H output (VIS/XMON copies are enqueued above when ntarget>0)
      CUDA_OK(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream));
      CUDA_OK(cudaStreamSynchronize(stream));

      // write output batch to ring
      if(write_exact(out_hdu->data_block, (const char*)h_out, out_bytes) <= 0){
        multilog(log, LOG_ERR, "write error/EOD on output %s (ipcio_write)\n", out_key);
        goto run_fail;
      }

      out_used = 1;

      // advance data-time (used for --ra/--dec tracking)
      frames_total += (uint64_t)batch_frames;

      // accumulate VIS/XMON per target
      if(ntarget > 0){
        for(size_t ti=0; ti<ntarget; ti++){
          auto &t = dump_targets[ti];
          for(int k=0;k<nfft;k++){
            const cf32 &v = vis_frame[ti*nfft + k];
            t.vis_acc[k].re += v.re;
            t.vis_acc[k].im += v.im;
            t.p0_acc[k] += p0_frame[ti*nfft + k];
            t.p1_acc[k] += p1_frame[ti*nfft + k];
          }
          t.frames_acc += (uint64_t)batch_frames;
        }
      }

      double t_now = now_sec();
      if(ntarget > 0 && (t_now - t_last_mon) >= monitor_sec){

        for(size_t ti=0; ti<ntarget; ti++){
          auto &t = dump_targets[ti];

          // coherence average across channels
          double coh_sum = 0.0;
          int coh_cnt = 0;
          for(int k=0;k<nfft;k++){
            double denom = sqrt(t.p0_acc[k]*t.p1_acc[k]) + 1e-30;
            double vabs = sqrt((double)t.vis_acc[k].re*(double)t.vis_acc[k].re + (double)t.vis_acc[k].im*(double)t.vis_acc[k].im);
            double coh = vabs / denom;
            coh_sum += coh; coh_cnt++;
          }
          float coh_avg = (coh_cnt>0) ? (float)(coh_sum/(double)coh_cnt) : 0.0f;

          // residual delay estimate: weighted linear fit of unwrapped phase vs fbin
          std::vector<double> ph_u((size_t)nfft);
          for(int k=0;k<nfft;k++){
            ph_u[k] = atan2((double)t.vis_acc[k].im, (double)t.vis_acc[k].re);
            if(k>0){
              while(ph_u[k]-ph_u[k-1] > M_PI) ph_u[k]-=2*M_PI;
              while(ph_u[k]-ph_u[k-1] < -M_PI) ph_u[k]+=2*M_PI;
            }
          }
          double Sw=0,Sf=0,Sp=0,Sff=0,Sfp=0;
          for(int k=0;k<nfft;k++){
            double w = sqrt(t.p0_acc[k]*t.p1_acc[k]) + 1e-30;
            double f = (double)fbin[k];
            double p = ph_u[k];
            Sw += w; Sf += w*f; Sp += w*p; Sff += w*f*f; Sfp += w*f*p;
          }
          double denom = (Sw*Sff - Sf*Sf);
          double a = (fabs(denom)>0) ? ((Sw*Sfp - Sf*Sp)/denom) : 0.0; // rad/Hz
          double tau_res = -a/(2*M_PI);

          // Snapshot for tracking CSV (and for external monitoring)
          double p0_sum=0.0, p1_sum=0.0;
          for(int k=0;k<nfft;k++){ p0_sum += t.p0_acc[k]; p1_sum += t.p1_acc[k]; }
          double norm = (double)(t.frames_acc) * (double)nspec;
          double p0_mean = (norm>0) ? ((p0_sum/(double)nfft)/norm) : 0.0;
          double p1_mean = (norm>0) ? ((p1_sum/(double)nfft)/norm) : 0.0;
          // SNR estimates (coherence-based)
          int nchan_valid = 0;
          for(int k=0;k<nfft;k++) if(t.p0_acc[k] > 0.0 && t.p1_acc[k] > 0.0) nchan_valid++;
          double df_hz = fs_hz / (double)nfft;
          double bw_eff_hz = (double)nchan_valid * df_hz;
          if(bw_eff_hz <= 0) bw_eff_hz = df_hz;
          double tint_emp = ((double)t.frames_acc * (double)FRAME_SAMP) / fs_hz;
          double tint_nom = monitor_sec;
          float snr_rad = coh_avg * (float)sqrt(2.0 * bw_eff_hz * tint_nom);
          float snr_emp = coh_avg * (float)sqrt(2.0 * bw_eff_hz * tint_emp);

          t.have_last = 1;
          t.last_coh_avg = coh_avg;
          t.last_snr_rad = snr_rad;
          t.last_snr_emp = snr_emp;
          t.last_bw_eff_hz = bw_eff_hz;
          t.last_tint_emp = tint_emp;
          t.last_tau_res = tau_res;
          t.last_p0_mean = p0_mean;
          t.last_p1_mean = p1_mean;
          t.last_t_mon = t_now;

          if(monitor && (monitor_all || (t.bi==dump_bi && t.bj==dump_bj && t.pol_code==dump_pol_code))){
            fprintf(stderr,"[XMON] dt=%.3fs frames=%llu (batch=%d) coh_avg=%.3f tau_res=%.3f ns baseline=%d-%d pol=%s\n",
                    (t_now - t0), (unsigned long long)t.frames_acc, batch_frames, coh_avg, tau_res*1e9, t.bi, t.bj, polcode_to_str(t.pol_code));
          }

          if(t.fp){
            write_visdump_record(t.fp, t_now, coh_avg, t.vis_acc.data(), t.p0_acc.data(), t.p1_acc.data(), (uint32_t)nfft);
          }

          // reset for next tick
          std::fill(t.vis_acc.begin(), t.vis_acc.end(), cf32{0.0f,0.0f});
          std::fill(t.p0_acc.begin(), t.p0_acc.end(), 0.0);
          std::fill(t.p1_acc.begin(), t.p1_acc.end(), 0.0);
          t.frames_acc = 0;
        }

        t_last_mon = t_now;
      }


    }  // end inner while(!g_stop) batches
run_ok:
    if(fp_track_csv){ fclose(fp_track_csv); fp_track_csv = nullptr; }
    if(out_used) ipcio_close(out_hdu->data_block);
    if(out_locked) dada_hdu_unlock_write(out_hdu);
    for(int a=0;a<NANT;a++){
      if(!antmask[a]) continue;
      if(in_used[a]) ipcio_close(in_hdu[a]->data_block);
      if(in_locked[a]) dada_hdu_unlock_read(in_hdu[a]);
    }
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "] RUN ended. waiting next headers...\n");
    continue;

run_fail:
    if(fp_track_csv){ fclose(fp_track_csv); fp_track_csv = nullptr; }
    // best-effort cleanup (close only if used)
    if(out_used) ipcio_close(out_hdu->data_block);
    if(out_locked) dada_hdu_unlock_write(out_hdu);
    for(int a=0;a<NANT;a++){
      if(!antmask[a]) continue;
      if(in_used[a]) ipcio_close(in_hdu[a]->data_block);
      if(in_locked[a]) dada_hdu_unlock_read(in_hdu[a]);
    }
    fprintf(stderr,"[bf-cuda " BF_CUDA_VERSION "][ERR] run failed. waiting next headers...\n");
    sleep(1);
    continue;
  }

  // cleanup
  for(auto &t : dump_targets){
    if(t.fp){ fclose(t.fp); t.fp = nullptr; }
  }
  CUFFT_OK(cufftDestroy(plan_fwd));
  CUFFT_OK(cufftDestroy(plan_inv));
  CUDA_OK(cudaStreamDestroy(stream));

  CUDA_OK(cudaFree(d_in));
  CUDA_OK(cudaFree(d_out));
  CUDA_OK(cudaFree(d_fft));
  CUDA_OK(cudaFree(d_beam));
  CUDA_OK(cudaFree(d_ph));
  if(d_cal) CUDA_OK(cudaFree(d_cal));
  CUDA_OK(cudaFree(d_coarse));
  CUDA_OK(cudaFree(d_vis));
  CUDA_OK(cudaFree(d_p0));
  CUDA_OK(cudaFree(d_p1));

  CUDA_OK(cudaFreeHost(h_in));
  CUDA_OK(cudaFreeHost(h_out));

  multilog_close(log);

  return 0;
}
