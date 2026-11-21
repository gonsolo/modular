"""Portable FastDiv correctness, run through the HAL DeviceContext.

Bucket A, item 2 of the completeness plan. This is the portable *purpose* of the
vendor-specific `basics/test_fast_div_ptx` (which hardcodes
`get_gpu_target["sm_90"]()` and checks the NVIDIA PTX string -- a portability
anti-pattern that can never run on Intel). Here we exercise the portable
`std.utils.fast_div.FastDiv` primitive *for its values*: on the B60, verify
`x / FastDiv[dtype](divisor) == x // divisor` for a range of `x`, for both a
power-of-2 (4) and a non-power-of-2 (3) divisor, on uint32 and uint64.

Only deps on the HAL `max` + `std` (minimal-HAL style, like our other tests).
Green on the B60 proves the portable FastDiv value path works on the
Intel/Level Zero target.

Intel/SPIR-V limitation found (2026-11-15): `FastDiv[.uint64]` does NOT build on
intel:xe2 -- the 64-bit algorithm uses a 128-bit wide intermediate
(`_uint_type_of_width[bit_width * 2]`), which SPIR-V cannot represent without the
`SPV_ALTERA_arbitrary_precision_integers` extension (not enabled). The uint32 path
uses only 64-bit wide intermediates and works. So this test covers uint32 (divisors
3 and 4); the uint64 case is documented, not exercised.

Run via Bazel:  //max/driver/src/max_license/test:hal_test_fast_div.mojo.test
"""

from std.utils.fast_div import FastDiv
from std.testing import assert_equal

from max.gpu import global_idx
from max.gpu.host import DeviceContext


# --- Kernel: x / FastDiv[dtype](divisor) -> out[tid] ------------------------
# Portable: `FastDiv[dtype](divisor)` is comptime; the division is the
# `__rdiv__` operator (`scalar / fast_div`). We rebind the input element to the
# FastDiv's unsigned type (matching the stock kernel), divide, and cast back.

def fast_div_kernel[
    dtype: DType,
    divisor: Int,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    len_dev: Int32,
):
    comptime fast_div = FastDiv[dtype](divisor)
    var tid = global_idx.x
    var length = Int(len_dev)
    if tid >= length:
        return
    var x = input[tid]
    var result = rebind[Scalar[fast_div.uint_type]](x) / fast_div
    output[tid] = result.cast[dtype]()


# --- Host: verify x / fastdiv == x // divisor over a range ------------------

def _run_fast_div[dtype: DType, divisor: Int](ctx: DeviceContext) raises:
    comptime length = 256
    comptime block_dim = 32
    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            # Spread across values incl. 0, small, and > divisor. Use i*7+1 to
            # avoid a degenerate all-zero pattern; cast to the unsigned dtype.
            in_host[i] = Scalar[dtype](UInt32(i) * 7 + 1)
    comptime kernel = fast_div_kernel[dtype=dtype, divisor=divisor]
    ctx.enqueue_function[kernel](
        out_device,
        in_device,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )
    with out_device.map_to_host() as out_host:
        for i in range(length):
            var x = UInt32(i) * 7 + 1
            var expected = Scalar[dtype](x // UInt32(divisor))
            assert_equal(
                out_host[i],
                expected,
                msg=String(
                    t"dtype={dtype} div={divisor} out[{i}]={out_host[i]} expected {expected} (x={x})"
                ),
            )


def main() raises:
    with DeviceContext() as ctx:
        # uint32, power-of-2 divisor
        _run_fast_div[.uint32, 4](ctx)
        # uint32, non-power-of-2 divisor
        _run_fast_div[.uint32, 3](ctx)
        # uint64 is NOT exercised: FastDiv[.uint64] needs a 128-bit wide
        # intermediate that SPIR-V can't represent without
        # SPV_ALTERA_arbitrary_precision_integers (see module docstring).
        # _run_fast_div[.uint64, 3](ctx)
