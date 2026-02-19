#!/usr/bin/env bash
set -euo pipefail

# build_bf_cuda_v1.5.2.sh
# Builds bf_fxbv_stream_dumpvis_cuda_v1.5.2.cu with nvcc and links/copies to a stable name.

PSRDADA_INC=${PSRDADA_INC:-/opt/linux_64/include}
PSRDADA_LIB=${PSRDADA_LIB:-/opt/linux_64/lib}

# SOFA C library (IAU). By default assume it is installed alongside PSRDADA.
SOFA_INC=${SOFA_INC:-${PSRDADA_INC}}
SOFA_LIB=${SOFA_LIB:-${PSRDADA_LIB}}
SOFA_LIBNAME=${SOFA_LIBNAME:-sofa_c}   # common name: -lsofa_c (override if yours differs)

ARCH=${ARCH:-sm_89}

SRC=${SRC:-bf_fxbv_stream_dumpvis_cuda_v1.5.2.cu}
OUT=${OUT:-bf_fxbv_stream_dumpvis_cuda_v1.5.2}
LINK=${LINK:-bf_fxbv_stream_dumpvis_cuda}

echo "[build] SRC=${SRC}"
echo "[build] OUT=${OUT}"
echo "[build] ARCH=${ARCH}"
echo "[build] PSRDADA_INC=${PSRDADA_INC}"
echo "[build] PSRDADA_LIB=${PSRDADA_LIB}"
echo "[build] SOFA_INC=${SOFA_INC}"
echo "[build] SOFA_LIB=${SOFA_LIB}"
echo "[build] SOFA_LIBNAME=${SOFA_LIBNAME}"

nvcc -O3 -std=c++14 -arch=${ARCH} -o ${OUT} ${SRC} \
  -lcufft -lm -lpthread \
  -I ${PSRDADA_INC} -I ${SOFA_INC} \
  -L ${PSRDADA_LIB} -lpsrdada \
  -L ${SOFA_LIB} -l${SOFA_LIBNAME}

# Create/refresh a stable name for convenience
if ln -sf "${OUT}" "${LINK}" 2>/dev/null; then
  echo "[OK] linked ${LINK} -> ${OUT}"
else
  cp -f "${OUT}" "${LINK}"
  echo "[OK] copied ${OUT} to ${LINK}"
fi

echo "[OK] built ${OUT}"
