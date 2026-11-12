#!/bin/bash
# intel_gpu_driver_build.sh <src> <prefix> [jobs]: builds IGC, gmmlib and the compute runtime (NEO) from the
# unpacked, patched sources in <src> and installs them into <prefix> (lib/: libigc, libigdfcl, libigdgmm,
# libze_intel_gpu). Used by the intel_gpu_driver repository rule; needs cmake, ninja, a C++ compiler, python3,
# bison and flex on the host.
set -euo pipefail
S=$1
P=$2
J=${3:-$(nproc)}
LOG=$P/build.log
mkdir -p "$P"
: > "$LOG"
step() { echo "intel_gpu_driver: $*" | tee -a "$LOG"; }
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_LIBDIR=lib)
# Modular's bazelrc sets CC=false and CXX=false so repository rules cannot reach host tools; this build needs them.
[ "${CC:-}" = false ] && unset CC
[ "${CXX:-}" = false ] && unset CXX

step "IGC (with LLVM, clang, lld, SPIR-V translator and tools from source)"
# -Bsymbolic and no semantic interposition keep IGC's own LLVM from binding to another LLVM in the process.
cmake -S "$S/igc" -B "$S/build-igc" "${common[@]}" \
  -DCMAKE_C_FLAGS=-fno-semantic-interposition -DCMAKE_CXX_FLAGS=-fno-semantic-interposition \
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-Bsymbolic \
  -DCCLANG_FROM_SYSTEM=OFF -DIGC_OPTION__CLANG_MODE=Source -DIGC_OPTION__LLD_MODE=Source \
  -DIGC_OPTION__LLVM_MODE=Source -DIGC_OPTION__LLVM_PREFERRED_VERSION=22.1.8 \
  -DIGC_OPTION__LINK_KHRONOS_SPIRV_TRANSLATOR=ON -DIGC_OPTION__SPIRV_TOOLS_MODE=Source \
  -DIGC_OPTION__USE_PREINSTALLED_SPIRV_HEADERS=OFF -DIGC_OPTION__VC_INTRINSICS_MODE=Source >> "$LOG" 2>&1
ninja -C "$S/build-igc" -j "$J" install >> "$LOG" 2>&1

step "gmmlib"
cmake -S "$S/gmmlib" -B "$S/build-gmmlib" "${common[@]}" >> "$LOG" 2>&1
ninja -C "$S/build-gmmlib" -j "$J" install >> "$LOG" 2>&1

step "compute runtime (Level Zero and OpenCL driver)"
PKG_CONFIG_PATH="$P/lib/pkgconfig" cmake -S "$S/compute-runtime" -B "$S/build-neo" "${common[@]}" \
  -DCMAKE_PREFIX_PATH="$P" -DNEO_SKIP_UNIT_TESTS=1 -DNEO_SKIP_MT_UNIT_TESTS=1 \
  -DIGC_DIR="$P" -DGMM_DIR="$P" -DOCL_ICD_VENDORDIR="$P/etc/OpenCL/vendors" >> "$LOG" 2>&1
ninja -C "$S/build-neo" -j "$J" install >> "$LOG" 2>&1
step "done"
