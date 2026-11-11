# Mojo on Intel GPUs (experimental)

This branch adds an Intel GPU target to Mojo: kernels compile to SPIR-V through
LLVM's SPIR-V backend and run through a Level Zero plugin for the MAX driver
(HAL). It is tested on one card, an Intel Arc B580 (Xe2, "Battlemage"), on Linux.

## What works

- `--target-accelerator intel:xe2`: kernels become SPIR-V 1.4 for OpenCL
  environments (`spirv64-unknown-opencl`), with sub-groups of 16 (the Mojo warp
  size on this target), shared memory, barriers, atomics, shuffles and ballots.
- `DeviceContext()` picks the B580 through the Level Zero plugin: buffers,
  copies, kernel launches and synchronization.
- A whole application: [Gonzales](https://github.com/gonsolo/Gonzales), a
  spectral path tracer written in Mojo, compiles and runs. Its Cornell box
  matches the NVIDIA render to 0.15%, and it renders Disney's Moana Island
  (using its own bridge to the Xe2 ray-tracing hardware, not part of this
  branch).

Not tested: the MAX graph compiler and the kernels library, other Intel GPUs.

## Commits

1. A SPIR-V GPU target for Intel GPUs (compiler).
2. stdlib: the Level Zero plugin and `is_intel_gpu`.
3. max: the Level Zero plugin and the `intel:xe2` target.
4. LLVM: SPIR-V backend fixes carried until they land upstream.
5. SPIR-V backend: Intel extensions, SIMD16, sub-group shuffle and ballot.
6. The Level Zero HAL plugin (`Target/lib/LevelZero`).
7. Optional debugging switches: `MODULAR_L0_SYNC_LAUNCH=1` (wait after every
   launch and copy), `MODULAR_L0_KERNEL_TIMES=1` (time per kernel),
   `MODULAR_L0_KERNEL_INFO=1` (private and spill memory per kernel).
8. The plugins are named after Level Zero, and the target after the Xe version
   (`intel:xe2`), which the device reports from its graphics IP version.
9. `@intel_gpu_driver`: IGC and the compute runtime from pinned, patched sources.

## Requirements

- Linux with the Level Zero loader.
- Intel's graphics compiler (IGC) and compute runtime with the fixes the target
  still needs. `@intel_gpu_driver` (opt-in, `bazel/intel_gpu_driver.bzl`) fetches
  them at pinned revisions, applies the patches in `bazel/public-patches/` and
  builds them (about an hour the first time; needs cmake, ninja, a C++ compiler,
  python3, bison and flex):

  ```sh
  ./bazelw --output_base=$HOME/bazel_hal build @intel_gpu_driver//:driver
  source $HOME/bazel_hal/external/+intel_gpu_driver+intel_gpu_driver/env.sh
  ```

  IGC is pinned to the master commit the fixes are tested on, because release
  v2.42.6 crashes compiling large kernels; it moves to the next release. The
  fixes: IGC [#460](https://github.com/intel/intel-graphics-compiler/pull/460),
  [#463](https://github.com/intel/intel-graphics-compiler/pull/463),
  [#465](https://github.com/intel/intel-graphics-compiler/pull/465),
  [#467](https://github.com/intel/intel-graphics-compiler/pull/467),
  [#468](https://github.com/intel/intel-graphics-compiler/pull/468) and two more
  in `bazel/public-patches/igc-*.patch`, and compute runtime
  [#1037](https://github.com/intel/compute-runtime/pull/1037). Small kernels also
  run on the stock driver.

The LLVM fixes are carried as patches in `bazel/public-patches/llvm-spirv-*.patch`:
[llvm#227599](https://github.com/llvm/llvm-project/pull/227599) (merged after the
pinned LLVM),
[#236368](https://github.com/llvm/llvm-project/pull/236368),
[#236369](https://github.com/llvm/llvm-project/pull/236369),
[#236370](https://github.com/llvm/llvm-project/pull/236370),
[#237724](https://github.com/llvm/llvm-project/pull/237724),
[#241166](https://github.com/llvm/llvm-project/pull/241166),
[#241167](https://github.com/llvm/llvm-project/pull/241167),
[#241168](https://github.com/llvm/llvm-project/pull/241168) and
[#241169](https://github.com/llvm/llvm-project/pull/241169).

## Build

The plugin needs the HAL device context, which a Bazel flag selects:

```sh
./bazelw --output_base=$HOME/bazel_hal build -c opt --//:hal_device_context=true \
  //Mojo/tools/mojo:mojo-full //Mojo/stdlib/std:std //max/driver/src/max_license:max \
  //Target:CPUPlugin.so //Target:LevelZeroPlugin.so
```

Then make a toolchain from a pip-installed `modular` package with the built
pieces swapped in:

```sh
R=$HOME/mojo-intel                      # the new toolchain
B=$HOME/bazel_hal/execroot/_main/bazel-out/k8-opt/bin
cp -a <venv>/lib/python3.*/site-packages/modular $R
rm -f $R/lib/libAsyncRTMojoBindings.so
cp -L $B/Mojo/tools/mojo/mojo-full $R/bin/mojo
cp -L $B/AsyncRT/libAsyncRTRuntimeGlobals.so $B/Mojo/libKGENCompilerRTShared.so \
      $B/Support/libMSupportGlobals.so $R/lib/
cp -L $B/Mojo/stdlib/std/std.mojoc $B/max/driver/src/max_license/max.mojoc $R/lib/mojo/
mkdir -p $R/lib/plugins
cp -L $B/Target/CPUPlugin.so $B/Target/LevelZeroPlugin.so $R/lib/plugins/
```

## Run

```sh
export MODULAR_MAX_PACKAGE_ROOT=$R MODULAR_MOJO_MAX_PACKAGE_ROOT=$R \
       MODULAR_MOJO_MAX_DRIVER_PATH=$R/bin/mojo MODULAR_MOJO_MAX_IMPORT_PATH=$R/lib/mojo
export MODULAR_DRIVER_PLUGINS="level_zero@$R/lib/plugins/LevelZeroPlugin.so;cpu@$R/lib/plugins/CPUPlugin.so"
$R/bin/mojo run --target-accelerator intel:xe2 vadd.mojo
```

`vadd.mojo`:

```mojo
from max.gpu import global_idx
from max.gpu.host import DeviceContext


def vadd(a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
         c: MutPointer[Float32, MutAnyOrigin], n: Int64):
    var i = global_idx.x
    if Int64(i) < n:
        c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def main() raises:
    comptime N = 1 << 20
    var ctx = DeviceContext()
    print("device:", ctx.name())
    var a = ctx.enqueue_create_host_buffer[DType.float32](N)
    var b = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(N):
        a[i] = Float32(i)
        b[i] = Float32(2 * i)
    var da = ctx.enqueue_create_buffer[DType.float32](N)
    var db = ctx.enqueue_create_buffer[DType.float32](N)
    var dc = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(da, a)
    ctx.enqueue_copy(db, b)
    ctx.enqueue_function[vadd](da.unsafe_ptr(), db.unsafe_ptr(), dc.unsafe_ptr(), Int64(N),
                               grid_dim=N // 256, block_dim=256)
    var c = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(c, dc)
    ctx.synchronize()
    var wrong = 0
    for i in range(N):
        if c[i] != Float32(3 * i):
            wrong += 1
    print("wrong:", wrong)
```

It prints `device: Intel(R) Arc(TM) B580 Graphics` and `wrong: 0`.

## Known limits

- Kernel arguments of type `Int`/`UInt` are rejected by the HAL compatibility
  layer; pass `Int64`.
- GPU `print` is not supported.
- Large kernels can need several hundred MiB of private memory each; Level Zero
  reserves it per kernel, so big applications can run out of VRAM
  ([compute-runtime#1037](https://github.com/intel/compute-runtime/pull/1037)
  proposes sharing one pool).
