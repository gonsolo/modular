"""Multi-warp strided lane-group reduction — generic reference (B60, 16-lane warp).

Step 1 of the multiwarp-reduce plan. Reduces a strided group whose span EXCEEDS
WARP_SIZE (num_lanes=4, stride=8 -> span 24 >= 16) to a single value, broadcast to
every member, by:

  Pass 1 (in-warp): each warp sums the group-members it holds, via a within-warp
      shuffle (safe: within-warp shuffles of locals are correct).
  Stage: write each warp's per-group partial to TRUE local (shared) memory.
  barrier(): work-group barrier (orders local-memory writes across warps).
  Pass 2 (merge): each lane sums its group's cross-warp partials from local mem.

CRITICAL (root-caused 2026-11-14): the staging buffer MUST be true local memory
via `external_memory[.SHARED]` (dynamic shared memory, sized at launch with
shared_mem_bytes). A GLOBAL buffer does NOT work: `barrier()` (work-group) only
orders local-memory writes, so cross-warp reads of a global buffer come back stale.

Run via Bazel:  //max/driver/src/max_license/test:proto_multiwarp.mojo.test
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal


def kernel(data: MutPointer[Scalar[.float32], MutAnyOrigin]):
    # num_lanes=4, stride=8, W=16, 2 warps (32-lane block).
    comptime W = 16
    comptime stride = 8
    comptime num_warps = 2
    comptime num_lanes = 4

    # TRUE local (shared) memory, indexed by GLOBAL lane (one slot per lane).
    # STATIC shared memory (unsafe_stack_allocation[.SHARED]) — the dynamic
    # external_memory[.SHARED] path is Unimplemented on this IGC build.
    # Size = W * num_warps = 32.
    var sram = unsafe_stack_allocation[
        W * num_warps, Float32, address_space=.SHARED
    ]()

    var lane = Int(thread_idx.x)
    var group = lane % stride

    # Stage ALL raw values to LOCAL memory (indexed by global lane).
    var val = data.unsafe_load[width=1](thread_idx.x)
    sram[lane] = val[0]
    barrier()

    # Sum this lane's group members by READING LOCAL memory (no shuffles at all —
    # sidesteps the IGC local-cross-warp-shuffle bug, §2.4b, by construction).
    # group members = {group, group+stride, group+2*stride, group+3*stride}.
    var total: Float32 = 0.0
    for k in range(num_lanes):
        var member = group + k * stride
        if member < W * num_warps:
            total = total + sram[member]

    # In-place store of the broadcast group result.
    data.unsafe_store(thread_idx.x, SIMD[.float32, 1](total))


def main() raises:
    comptime W = 16
    comptime num_warps = 2
    comptime block_size = W * num_warps
    comptime num_lanes = 4
    comptime stride = 8

    with DeviceContext() as ctx:
        # ground truth: value at lane i = i ; each strided group's sum broadcast
        var data = ctx.enqueue_create_host_buffer[.float32](block_size)
        for i in range(block_size):
            data[i] = Float32(i)

        var expected = ctx.enqueue_create_host_buffer[.float32](block_size)
        for i in range(block_size):
            var base = i % stride
            var s: Float32 = 0.0
            for k in range(num_lanes):
                var member = base + k * stride
                if member < block_size:
                    s += data[member]
            expected[i] = s

        var data_buf = ctx.enqueue_create_buffer[.float32](block_size)
        ctx.enqueue_copy(data_buf, data)

        ctx.enqueue_function[kernel](
            data_buf,
            grid_dim=1,
            block_dim=block_size,
        )
        ctx.synchronize()
        ctx.enqueue_copy(data, data_buf)

        # every lane must now hold its group's sum
        for i in range(block_size):
            assert_equal(data[i], expected[i])
