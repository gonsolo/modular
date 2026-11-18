"""Portable prefetch, run through the HAL DeviceContext.

Bucket A, item 4 of the completeness plan. The portable *purpose* of the vendor
prefetch tests (which assert on the NVIDIA PTX `prefetch` asm string) is "the
prefetch op works on the target." Here we verify the portable
`std.sys.intrinsics.prefetch` op BUILDS and RUNS correctly on the B60 (the
prefetched data is read back correctly). We skip the PTX-asm-string assertion
(vendor "how"); the point is that the prefetch *operation* works on the
Intel/Level Zero target.

Background (a gap we worked around): out of the box, `prefetch` on the Intel
target lowers to `llvm.prefetch` -> SPIR-V's OCL `__spirv_ocl_prefetch`, which
the IGC backend does not provide a definition for (undefined reference at the
device link). A prefetch is only a *hint* (correctness does not depend on it),
so the Level Zero *std* plugin supplies a no-op `prefetch_fn` (see
`Mojo/stdlib/std/_plugin/level_zero/level_zero_plugin.mojo`), which makes the
portable `prefetch` op build + run here. Replace the no-op with a real
IGC-supported cache hint once the backend exposes one.

Only deps on the HAL `max` + `std` (minimal-HAL style, like our other tests).

Run via Bazel:  //max/driver/src/max_license/test:hal_test_prefetch.mojo.test
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
