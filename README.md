![cbfv](fig_bf.png)
![cbfv](fig_nobf.png)
![dada_m](dada_monitor0.png)
![cpu](cpuhtop.png)
![gpu](gpu_1.png)

![bf_spec](bf_spec.png)
![vdif_2bit_baseband](bf_2bit_1.png)
![vdif_2bit_m5spec](bf_vdif_2bit.png)
![m5specplot](plt.pdf)

# CUDA 4-Antenna Beamformer Processing Pipeline

PSRDADA input rings: a000,a002,a004,a006
| read_exact() per ant (L1706-L1724)
V
Host pinned h_in (int16)
shape: [a=4, f=batch, 4096 int16]
| H->D cudaMemcpyAsync (L1836)
V
Device d_in (int16)
shape: [a=4, f=batch, 4096 int16]
| unpack_to_fft_input kernel (L805-L850)
(+ coarse shift if !no_delay, L832-L834)
int16 -> float2 (cufftComplex)
time shift by coarse[a] with zero-pad
V
Device d_fft (cufftComplex) time-domain blocks
shape: [a, p=2, f, sp, k]
where sp = 1024/nfft, k=nfft
total elems = 42batch1024
| cuFFT FORWARD plan_fwd (L1845)
F-engine channelization
V
Device d_fft (cufftComplex) freq-domain
| apply_weights kernel (L899-L1899)
multiply weights in freq-domain
use_geo = !no_delay (L1846)
use_cal = (d_cal!=nullptr) (L1847)
w = w_geo(a,k) * w_cal(a,p,k)
+--> (side branch) compute_vis_pow kernel (L908-L1018) for each target (L1853-L1863)
computes V_ij(k), P_i(k), P_j(k) on weighted spectra
D->H copies of vis/p0/p1 (L1859-L1861)
V
Device d_fft (weighted spectra)
| beamform_sum kernel (L901-L936)
B-engine: sum antennas -> beam spectrum
B_p(k) = (1/N_used) * Σ a X'_{a,p}(k)
V
Device d_beam (cufftComplex) freq-domain beam
shape: [p=2, f=batch, sp, k]
total elems = 2batch*1024
| cuFFT INVERSE plan_inv (L1872)
V-engine: beam spectrum -> time blocks
V
Device d_beam (time-domain blocks)
| pack_output kernel (L938-L978)
normalize 1/nfft + clip to int16
V
Device d_out (int16)
shape: [f=batch, 4096 int16]
| D->H cudaMemcpyAsync + cudaStreamSynchronize (L1879-L1880)
V
Host pinned h_out (int16)
| write_exact() to output ring b000 (L1882-L1886)
V
PSRDADA output ring: b000
=> beamformed complex voltage stream (ci16), two pols (X/Y), same frame format as input
