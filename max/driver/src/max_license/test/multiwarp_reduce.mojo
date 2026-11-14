"""`lane_group_reduce_multiwarp` — stdlib dispatch test (B60, 16-lane warp).

Step 2 of the multiwarp-reduce plan. Exercises the NEW stdlib function
`warp.lane_group_reduce_multiwarp` through its real dispatch, NOT an inline
prototype. Two cases:

  (1) span < WARP_SIZE  -> fast path, delegates to the existing in-warp
      butterfly `lane_group_reduce` (byte-for-byte the old behavior).
  (2) span >= WARP_SIZE -> dispatches to the plugin hook
      `MaxPluginHooks.strided_multiwarp_reduce_fn`. On the Level Zero / B60
      target no backend overrides that method, so its *default body* runs:
      the correct-by-generic reference (local-memory staging + work-group
      barrier + merge, no shuffles — immune to the IGC local-cross-warp-
      shuffle defect).

Run via Bazel:  //max/driver/src/max_license/test:multiwarp_reduce.mojo.test
"""

import max.gpu.primitives.warp as warp
from max.gpu import thread_idx
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from std.testing import assert_equal


# --- Kernel: each lane loads its value and runs the multiwarp reduce ---------
# num_lanes / stride / block_threads are comptime params (bound at launch).

def kernel_multiwarp[
    num_lanes: Int,
    stride: Int,
    block_threads: Int,
](data: MutPointer[Scalar[.float32], MutAnyOrigin]):
    # num_lanes/stride/block_threads are infer-only (comptime) template params,
    # so they are constants here and usable in the `[...]` position of the call.
    var lane = Int(thread_idx.x)
    var val = data.unsafe_load[width=1](lane)

    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    # The new stdlib function; dispatches on span vs WARP_SIZE.
    # val_type / simd_width / FuncType are inferred from `val` and `func`; the
    # explicit params (after `//`) are num_lanes, stride, block_threads.
    var result = warp.lane_group_reduce_multiwarp[
        num_lanes, stride, block_threads
    ](val, reduce_add)
    data.unsafe_store(lane, result)


# --- Host: launch + verify each lane holds its strided group's sum ----------

def _check[
    num_lanes: Int,
    stride: Int,
    block_size: Int,
](ctx: DeviceContext) raises:
    # num_lanes / stride / block_size are comptime (generic) params, so they may
    # be bound into the kernel. Bind them into a concrete kernel, then launch.
    comptime kernel = kernel_multiwarp[num_lanes, stride, block_size]

    var host = ctx.enqueue_create_host_buffer[.float32](block_size)
    for i in range(block_size):
        host[i] = Float32(i)

    var expected = ctx.enqueue_create_host_buffer[.float32](block_size)
    for i in range(block_size):
        var base = i % stride
        var s: Float32 = 0.0
        for k in range(num_lanes):
            var member = base + k * stride
            if member < block_size:
                s += host[member]
        expected[i] = s

    var d = ctx.enqueue_create_buffer[.float32](block_size)
    ctx.enqueue_copy(d, host)
    ctx.enqueue_function[kernel](
        d,
        grid_dim=1,
        block_dim=block_size,
    )
    ctx.synchronize()
    ctx.enqueue_copy(host, d)

    # Every lane must now hold its strided group's sum (broadcast).
    for i in range(block_size):
        assert_equal(host[i], expected[i])


def main() raises:
    comptime W = WARP_SIZE

    with DeviceContext() as ctx:
        # (1) Fast path: span < WARP_SIZE (delegates to in-warp butterfly).
        #     block = 1 warp (16), num_lanes=4, stride=4 -> span 12 < 16.
        _check[4, 4, W](ctx)
        #     num_lanes=8, stride=2 -> span 14 < 16 (still in-warp).
        _check[8, 2, W](ctx)

        # (2) Out-of-warp path: span >= WARP_SIZE (generic reference).
        #     block = 2 warps (32), num_lanes=4, stride=8 -> span 24 >= 16.
        _check[4, 8, 2 * W](ctx)
        #     block = 4 warps (64), num_lanes=8, stride=8 -> span 56 >= 16.
        _check[8, 8, 4 * W](ctx)
        #     block = 2 warps (32), num_lanes=16, stride=2 -> span 30 >= 16.
        _check[16, 2, 2 * W](ctx)
