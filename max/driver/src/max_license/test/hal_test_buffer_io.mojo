"""Portable vectorized / strided buffer load+store, run through the HAL DeviceContext.

Bucket A, item 1 of the completeness plan. This is the portable *purpose* of the
vendor-specific `memory/test_buffer_io` (which uses AMD's `AMDBufferResource`
descriptor -- a vendor *how*). Here we exercise the portable primitives directly:

  - `ptr.unsafe_load[width=N, alignment=A](offset)` /
    `unsafe_store[width=N, alignment=A](offset, val)` -- the portable vectorized
    load/store (the "move a vector of values in/out of VRAM" activity).
  - `strided_load[width](ptr, stride)` -- the portable strided gather.

Only deps on the HAL `max` + `std` (minimal-HAL style, like our other 13 tests).

Findings on the B60 (Intel/Level Zero, IGC 2.41.15):
  - scalar + vectorized (width=4) load/store: PASS.
  - `strided_load` with simd_width > 4: PASS -- via a workaround. The vectorized
    gather (`gather` intrinsic) lowers to a `G_STEP_VECTOR` that the SPIR-V/IGC
    GISel backend does not legalize (modular/modular#7620), so a width>1 gather
    fails to compile out of the box. The Level Zero *std* plugin sets
    `has_generic_address_space = False`, which routes `strided_load` through the
    scalar per-lane loop instead (correct, if not fastest). See the
    `strided_load_kernel` note below.

Run via Bazel:  //max/driver/src/max_license/test:hal_test_buffer_io.mojo.test
"""

from std.sys.intrinsics import strided_load
from std.testing import assert_equal

from max.gpu import global_idx
from max.gpu.host import DeviceContext


# --- Kernel 0: plain scalar load/store (pipeline sanity) ---------------------
# Each thread doubles its element. Confirms the launch/index/map_to_host
# pipeline before we test the vectorized primitives.

def scalar_double_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    len_dev: Int32,
):
    var length = Int(len_dev)
    var tid = global_idx.x
    if tid >= length:
        return
    output[tid] = 2 * input[tid]


# --- Kernel 1: vectorized load/store (the portable buffer_io purpose) --------
# Each thread loads a vector of `W` floats, doubles it, stores it back. Done with
# the portable `unsafe_load[width]`/`unsafe_store[width]` (not AMD's
# buffer-resource descriptor).

def vectorized_load_store_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    len_dev: Int32,
):
    comptime W = 4
    comptime align = 16 * W
    var tid = global_idx.x
    var length = Int(len_dev)
    var base = tid * W
    if base + W > length:
        return
    var v = input.unsafe_load[width=W, alignment=align](base)
    for i in range(W):
        v[i] = 2 * v[i]
    output.unsafe_store[width=W, alignment=align](base, v)


# --- Kernel 2: strided gather (the portable strided_load purpose) -----------
# Each thread gathers `W` elements spaced `S` apart via `strided_load[width]`.
#
# NOTE: `strided_load` with W > 1 lowers to the `gather` intrinsic, whose
# offset-vector construction (`G_STEP_VECTOR`) is not legalizable by the SPIR-V
# GISel backend (modular/modular#7620). The Level Zero *std* plugin therefore
# sets `has_generic_address_space = False`, routing `strided_load` through the
# scalar per-lane loop (correct, if not fastest) instead. So W = 4 builds + runs.

def strided_load_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    len_dev: Int32,
):
    comptime W = 4
    comptime S = 8
    var tid = global_idx.x
    var length = Int(len_dev)
    var base = tid * W
    # The gather reads input[base + k*S] for k in 0..W-1, so the max input index
    # is base + (W-1)*S. Guard on THAT (not just the output range) so the gather
    # never reads out of bounds.
    if base + (W - 1) * S >= length:
        return
    # `strided_load` reads starting at the pointer's address, so offset the
    # pointer by `base` first: lane k reads input[base + k*S].
    var v = strided_load[W](input.unsafe_offset(base), S)
    for i in range(W):
        output[base + i] = v[i]


# --- Host: verify -----------------------------------------------------------

def test_scalar_pipeline(ctx: DeviceContext) raises:
    comptime length = 1024
    comptime block_dim = 32
    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Float32(i + 1)
    comptime kernel = scalar_double_kernel
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
                msg=String(t"scalar out[{i}] = {out_host[i]} expected {Float32(2 * (i + 1))}"),
            )


def test_vectorized_load_store(ctx: DeviceContext) raises:
    comptime length = 1024
    comptime block_dim = 32
    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Float32(i + 1)
    comptime kernel = vectorized_load_store_kernel
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
                msg=String(t"vec out[{i}] = {out_host[i]} expected {Float32(2 * (i + 1))}"),
            )


def test_strided_load(ctx: DeviceContext) raises:
    comptime W = 4  # matches the kernel (vectorized gather; works via the
    # std-plugin has_generic_address_space=False scalar-loop route, see #7620)
    comptime S = 8
    comptime block_dim = 32
    comptime num_threads = 128  # grid=4 x block=32
    # The gather for thread t reads input[t*W + k*S] (k in 0..W-1); the max
    # input index is (num_threads-1)*W + (W-1)*S. Size the buffer to cover it.
    comptime length = (num_threads - 1) * W + (W - 1) * S + 1
    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Float32(i + 1)
    comptime kernel = strided_load_kernel
    ctx.enqueue_function[kernel](
        out_device,
        in_device,
        Int32(length),
        grid_dim=(num_threads // block_dim),
        block_dim=(block_dim),
    )
    with out_device.map_to_host() as out_host:
        # Every thread's gather is in-bounds by construction, so all of its W
        # output slots are valid: out[t*W + k] = input[t*W + k*S] = (t*W + k*S) + 1.
        for t in range(num_threads):
            for k in range(W):
                var idx = t * W + k
                var expected = Float32(t * W + k * S + 1)
                assert_equal(
                    out_host[idx],
                    expected,
                    msg=String(t"strided out[{idx}] = {out_host[idx]} expected {expected}"),
                )


def main() raises:
    with DeviceContext() as ctx:
        test_scalar_pipeline(ctx)
        test_vectorized_load_store(ctx)
        test_strided_load(ctx)
