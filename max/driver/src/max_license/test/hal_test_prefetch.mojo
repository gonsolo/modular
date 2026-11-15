"""Portable prefetch on the B60 -- DOCUMENTED GAP (does not build on intel:xe2).

Bucket A, item 4 of the completeness plan. The portable *purpose* of the vendor
prefetch tests (which assert on the NVIDIA PTX `prefetch` asm string) is "the
prefetch op works on the target." Finding on the B60: it does NOT build.

The portable `std.sys.intrinsics.prefetch` lowers, on the Intel target, to
SPIR-V's OCL `__spirv_ocl_prefetch` (the Level Zero plugin does not provide a
`prefetch_fn`, so `prefetch` falls back to `llvm.prefetch` -> `__spirv_ocl_prefetch`).
The link step then fails with an undefined reference:

    undefined reference to `_Z20__spirv_ocl_prefetchPU3AS4KPKcm'
    (__spirv_ocl_prefetch(char const* const AS4*, unsigned long))

i.e. the SPIR-V/IGC backend does not provide a definition for
`__spirv_ocl_prefetch` on this target. This is a distinct Intel/SPIR-V gap
(separate from the G_STEP_VECTOR gather gap, modular/modular#7620).

So this test is kept for documentation / as a canary (uncomment the BUILD entry
in BUILD.bazel to reproduce), but is NOT registered as an active hal_gpu_test --
it would break the suite. Re-enable once the backend provides
`__spirv_ocl_prefetch` (or the Level Zero plugin supplies a `prefetch_fn`).

Only deps on the HAL `max` + `std` (minimal-HAL style, like our other tests).
"""

from std.sys.intrinsics import prefetch
from std.testing import assert_equal

from max.gpu import global_idx
from max.gpu.host import DeviceContext


# --- Kernel: prefetch then read each element --------------------------------
# Each thread prefetches its element (a read hint into the data cache) and then
# reads it back, doubling it. The prefetch is a hint; the value read must still
# be correct. This proves the portable prefetch op builds + runs on the B60.

def prefetch_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    len_dev: Int32,
):
    var tid = global_idx.x
    var length = Int(len_dev)
    if tid >= length:
        return
    # Read-prefetch hint for this thread's element (data cache, high locality).
    # dtype is inferred from the pointer (the working stock test calls it bare).
    prefetch(input.unsafe_offset(tid))
    output[tid] = 2 * input[tid]


def main() raises:
    with DeviceContext() as ctx:
        comptime length = 1024
        comptime block_dim = 32
        var in_device = ctx.enqueue_create_buffer[.float32](length)
        var out_device = ctx.enqueue_create_buffer[.float32](length)
        with in_device.map_to_host() as in_host:
            for i in range(length):
                in_host[i] = Float32(i + 1)
        comptime kernel = prefetch_kernel
        ctx.enqueue_function[kernel](
            out_device,
            in_device,
            Int32(length),
            grid_dim=(length // block_dim),
            block_dim=(block_dim),
        )
        with out_device.map_to_host() as out_host:
            for i in range(length):
                assert_equal(
                    out_host[i],
                    Float32(2 * (i + 1)),
                    msg=String(t"prefetch out[{i}] = {out_host[i]} expected {Float32(2 * (i + 1))}"),
                )
