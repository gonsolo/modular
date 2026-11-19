"""Portable vectorized buffer load+store, run through the HAL DeviceContext.

Bucket A, item 1 of the completeness plan. This is the portable *purpose* of the
vendor-specific `memory/test_buffer_io` (which uses AMD's `AMDBufferResource`
descriptor -- a vendor *how*). Here we exercise the portable primitives directly:

  - `ptr.unsafe_load[width=N, alignment=A](offset)` /
    `unsafe_store[width=N, alignment=A](offset, val)` -- the portable vectorized
    load/store (the "move a vector of values in/out of VRAM" activity).

The portable *strided gather* (`strided_load[width](ptr, stride)`) is NOT exercised
here: a width>1 gather lowers to the `gather` intrinsic, whose offset-vector
construction (`G_STEP_VECTOR`) is not legalizable by the SPIR-V/IGC GISel backend
(modular/modular#7620). It will be added back with a note once the upstream fix lands.

Only deps on the HAL `max` + `std` (minimal-HAL style, like the other tests).

Findings on the B60 (Intel/Level Zero, IGC 2.41.15):
  - scalar + vectorized (width=4) load/store: PASS.

Run via Bazel:  //max/driver/src/max_license/test:hal_test_buffer_io.mojo.test
"""

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


# NOTE: the portable *strided gather* (`strided_load`) is not exercised here -- a
# width>1 gather lowers to `G_STEP_VECTOR`, not legalizable by the SPIR-V/IGC GISel
# backend (modular/modular#7620). It is added back with a note once that upstream
# fix lands. See the module docstring.


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


def main() raises:
    with DeviceContext() as ctx:
        test_scalar_pipeline(ctx)
        test_vectorized_load_store(ctx)
