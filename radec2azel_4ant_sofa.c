// radec2azel_4ant_sofa.c
// Use IAU SOFA (C) to convert ICRS RA/Dec -> observed Az/El for 4 antennas
// Antenna geodetic positions are read from CSV: id, lon_deg, lat_deg, elev_m (WGS84)

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "sofa.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI/180.0)
#define RAD2DEG (180.0/M_PI)

typedef struct {
    int id;
    double lon_deg;
    double lat_deg;
    double hm;   // meters
} ant_t;

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --ra HH:MM:SS.S --dec [+/-]DD:MM:SS.S [options]\n\n"
        "Options:\n"
        "  --ant  PATH        Antenna CSV (default: antpos_4ant_geo.csv)\n"
        "  --utc  STR         UTC time: YYYY-MM-DDTHH:MM:SS[.sss]\n"
        "                    also accepts 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD-HH:MM:SS'\n"
        "                    (default: current system UTC)\n"
        "  --dut1 SEC         UT1-UTC seconds (default: 0)\n"
        "  --xp   ARCSEC      polar motion x (arcsec, default: 0)\n"
        "  --yp   ARCSEC      polar motion y (arcsec, default: 0)\n"
        "  --phpa HPA         pressure (hPa). If 0 => no refraction (default: 0)\n"
        "  --tc   C           temperature Celsius (default: 0)\n"
        "  --rh   0-1         relative humidity (default: 0)\n"
        "  --wl   UM          wavelength micrometers (used only if phpa>0) (default: 0)\n"
        "  -h, --help         show this help\n\n"
        "Notes:\n"
        "  - RA/Dec assumed ICRS (J2000) coordinates.\n"
        "  - Output Az is measured from North through East (SOFA convention).\n",
        prog);
}

static int parse_ra_hms(const char *s, double *ra_rad)
{
    int h=0, m=0;
    double sec=0.0;

    if (sscanf(s, "%d:%d:%lf", &h, &m, &sec) != 3 &&
        sscanf(s, "%d %d %lf", &h, &m, &sec) != 3) {
        return -1;
    }

    // SOFA: time->angle, 24h = 2pi
    int st = iauTf2a('+', h, m, sec, ra_rad);
    return (st == 0) ? 0 : -2;
}

static int parse_dec_dms(const char *s, double *dec_rad)
{
    char sign = '+';
    int d=0, m=0;
    double sec=0.0;

    // with sign
    if (sscanf(s, " %c%d:%d:%lf", &sign, &d, &m, &sec) == 4 ||
        sscanf(s, " %c%d %d %lf",  &sign, &d, &m, &sec) == 4) {
        if (sign != '+' && sign != '-') sign = '+';
        int st = iauAf2a(sign, d, m, sec, dec_rad);
        return (st == 0) ? 0 : -2;
    }

    // without sign => '+'
    if (sscanf(s, "%d:%d:%lf", &d, &m, &sec) == 3 ||
        sscanf(s, "%d %d %lf",  &d, &m, &sec) == 3) {
        int st = iauAf2a('+', d, m, sec, dec_rad);
        return (st == 0) ? 0 : -2;
    }

    return -1;
}

static int parse_utc_string(const char *s, int *y,int *mo,int *d,int *hh,int *mm,double *ss)
{
    // Accept:
    // YYYY-MM-DDTHH:MM:SS(.sss)
    // YYYY-MM-DD HH:MM:SS(.sss)
    // YYYY-MM-DD-HH:MM:SS(.sss)
    if (sscanf(s, "%d-%d-%dT%d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return 0;
    if (sscanf(s, "%d-%d-%d %d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return 0;
    if (sscanf(s, "%d-%d-%d-%d:%d:%lf", y, mo, d, hh, mm, ss) == 6) return 0;
    return -1;
}

static int load_ant_csv(const char *path, ant_t *ants, int maxants)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen antenna csv");
        return -1;
    }

    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        // skip comments/blank
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;

        int id;
        double lon, lat, hm;
        if (sscanf(p, "%d,%lf,%lf,%lf", &id, &lon, &lat, &hm) == 4) {
            if (n < maxants) {
                ants[n].id = id;
                ants[n].lon_deg = lon;
                ants[n].lat_deg = lat;
                ants[n].hm = hm;
                n++;
            }
        }
    }
    fclose(fp);
    return n;
}

static void get_system_utc(int *y,int *mo,int *d,int *hh,int *mm,double *ss)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
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

int main(int argc, char **argv)
{
    const char *ra_s  = NULL;
    const char *dec_s = NULL;
    const char *ant_path = "antpos_4ant_geo.csv";
    const char *utc_s = NULL;

    double dut1 = 0.0;        // seconds
    double xp_arcsec = 0.0;   // arcsec
    double yp_arcsec = 0.0;   // arcsec

    double phpa = 0.0; // hPa, 0 => no refraction
    double tc   = 0.0; // Celsius
    double rh   = 0.0; // 0..1
    double wl   = 0.0; // micrometers

    for (int i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--ra") && i+1<argc) {
            ra_s = argv[++i];
        } else if (!strcmp(argv[i], "--dec") && i+1<argc) {
            dec_s = argv[++i];
        } else if (!strcmp(argv[i], "--ant") && i+1<argc) {
            ant_path = argv[++i];
        } else if (!strcmp(argv[i], "--utc") && i+1<argc) {
            utc_s = argv[++i];
        } else if (!strcmp(argv[i], "--dut1") && i+1<argc) {
            dut1 = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--xp") && i+1<argc) {
            xp_arcsec = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--yp") && i+1<argc) {
            yp_arcsec = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--phpa") && i+1<argc) {
            phpa = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--tc") && i+1<argc) {
            tc = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--rh") && i+1<argc) {
            rh = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--wl") && i+1<argc) {
            wl = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (!ra_s || !dec_s) {
        fprintf(stderr, "Error: --ra and --dec are required.\n");
        usage(argv[0]);
        return 2;
    }

    // Parse RA/Dec
    double rc=0.0, dc=0.0;
    int st_ra = parse_ra_hms(ra_s, &rc);
    int st_dc = parse_dec_dms(dec_s, &dc);
    if (st_ra) { fprintf(stderr, "Bad RA format: %s\n", ra_s); return 3; }
    if (st_dc) { fprintf(stderr, "Bad Dec format: %s\n", dec_s); return 3; }

    // Time -> UTC JD
    int y, mo, day, hh, mm;
    double sec;
    if (utc_s) {
        if (parse_utc_string(utc_s, &y,&mo,&day,&hh,&mm,&sec)) {
            fprintf(stderr, "Bad UTC format: %s\n", utc_s);
            return 4;
        }
    } else {
        get_system_utc(&y,&mo,&day,&hh,&mm,&sec);
    }

    double utc1=0.0, utc2=0.0;
    int st = iauDtf2d("UTC", y, mo, day, hh, mm, sec, &utc1, &utc2);
    if (st) {
        fprintf(stderr, "iauDtf2d failed, status=%d\n", st);
        return 5;
    }

    // Load antennas
    ant_t ants[16];
    int nants = load_ant_csv(ant_path, ants, 16);
    if (nants <= 0) {
        fprintf(stderr, "No antennas loaded from %s\n", ant_path);
        return 6;
    }

    // polar motion: arcsec -> rad
    double xp = xp_arcsec * (DEG2RAD/3600.0);
    double yp = yp_arcsec * (DEG2RAD/3600.0);

    // Output header
    printf("# Input ICRS: RA=%s  Dec=%s\n", ra_s, dec_s);
    printf("# UTC: %04d-%02d-%02d %02d:%02d:%09.6f  (dut1=%.6f s)\n",
           y,mo,day,hh,mm,sec,dut1);
    printf("# Antenna CSV: %s (loaded %d)\n", ant_path, nants);
    printf("# Columns: id  lon_deg  lat_deg  hm_m  az_deg  el_deg\n");

    for (int i=0; i<nants; i++) {
        double lon = ants[i].lon_deg * DEG2RAD; // east+
        double lat = ants[i].lat_deg * DEG2RAD;
        double hm  = ants[i].hm;

        double aob=0.0, zob=0.0, hob=0.0, dob=0.0, rob=0.0, eo=0.0;

        // pr,pd,px,rv set to zero for typical extragalactic radio sources
        iauAtco13(rc, dc,
                  0.0, 0.0, 0.0, 0.0,
                  utc1, utc2, dut1,
                  lon, lat, hm,
                  xp, yp,
                  phpa, tc, rh, wl,
                  &aob, &zob, &hob, &dob, &rob, &eo);

        double az_deg = fmod(aob * RAD2DEG + 360.0, 360.0);
        double el_deg = (M_PI/2.0 - zob) * RAD2DEG;

        printf("%d  %.8f  %.8f  %.3f  %.6f  %.6f\n",
               ants[i].id, ants[i].lon_deg, ants[i].lat_deg, hm, az_deg, el_deg);
    }

    return 0;
}
