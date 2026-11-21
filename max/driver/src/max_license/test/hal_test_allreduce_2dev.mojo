"""Bucket D, tier 3: a real 2-device host-staged allreduce on the dual-B60.

This drives the REAL `comm` library's non-P2P (host-staged) allreduce on this box's
two B60s -- the same path the stock `test_allreduce` exercises via its
`allreduce_naive_test()`. We re-implement that driver here in the minimal-HAL test
directory for two reasons:

  1. The stock `test_allreduce` is platform-gated to NVIDIA multi-GPU
     (`target_compatible_with = has_multi_gpu`), so it cannot be built on
     `intel:xe2`. The `//max:comm` *library* itself builds cleanly on Intel
     (validated), so we can call the real host-staged allreduce directly.
  2. Our two B60s are on separate PCIe switch ports -> no direct P2P
     (`can_access` == False, confirmed by hal_test_multidev_probe). The
     `comm` library therefore takes its **host-staged** fallback
     (`_allreduce_naive_single`), which stages each peer's input through
     host-visible `DeviceBuffer` copies. That is the path under test.

What it verifies (the portable *purpose* of the allreduce test): two devices each
hold an input; a 2-rank allreduce combines them so that **every** device ends up
with the sum of both inputs. Device 0 holds all 1s, device 1 holds all 2s -> the
allreduce result on BOTH devices must be all 3s.

Deps: the HAL `max` (for DeviceContext/DeviceBuffer), `std`, plus `//max:comm` and
`//max:layout` (the real collective + tensor primitives). Run with
`ZE_AFFINITY_MASK` unset (hal_gpu_multi_test) so both B60s are visible.

Run via Bazel:  //max/driver/src/max_license/test:hal_test_allreduce_2dev.mojo.test
"""

from std.collections import Array, List
from std.memory.alloc import unsafe_alloc
from std.testing import assert_almost_equal, assert_true
from std.utils import StaticTuple

from layout import Coord, TileTensor, row_major

from comm.allreduce import _allreduce_naive_single

from max.gpu.host import DeviceBuffer, DeviceContext


def main() raises:
    comptime ngpus = 2
    comptime length = 8 * 1024

    # Both B60s must be visible (affinity mask dropped by hal_gpu_multi_test).
    assert_true(
        DeviceContext.number_of_devices() >= ngpus,
        "expected >= 2 devices for the 2-rank allreduce",
    )

    # --- One context + in/out buffer per device. -------------------------------
    var ctxs = List[DeviceContext]()
    for i in range(ngpus):
        ctxs.append(DeviceContext(device_id=i))

    var in_dev = List[DeviceBuffer[.float32]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[.float32]](capacity=ngpus)
    var host_ptrs = List[MutPointer[Float32, MutUntrackedOrigin]](capacity=ngpus)

    for i in range(ngpus):
        in_dev.append(ctxs[i].enqueue_create_buffer[.float32](length))
        out_dev.append(ctxs[i].enqueue_create_buffer[.float32](length))
        var h = unsafe_alloc[Float32](length)
        host_ptrs.append(h)
        # Device i holds all (i+1): device 0 -> 1s, device 1 -> 2s.
        for j in range(length):
            h[j] = Float32(i + 1)
        ctxs[i].enqueue_copy(in_dev[i], host_ptrs[i])

    # --- Build the TileTensor views (pointer-backed, host-visible). ------------
    comptime InTensorType = TileTensor[
        .float32, type_of(row_major(length)), ImmutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: TileTensor(
            in_dev[i].unsafe_ptr().as_imm(), row_major(length)
        )
    )

    comptime OutTensorType = TileTensor[
        .float32, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {
            mut out_dev, imm
        } -> OutTensorType: TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()
    )

    # The epilogue lambda captures a StaticTuple of the out TileTensors (NOT the
    # List of DeviceBuffers, which is not implicitly copyable) and stores into
    # tensor i.
    var out_tensors_capture = StaticTuple[OutTensorType, ngpus]()
    for i in range(ngpus):
        out_tensors_capture[i] = TileTensor(out_dev[i], row_major(length))

    # --- Run the REAL host-staged allreduce, one instance per device. ----------
    # Each device stages BOTH inputs (its own + the peer's, through host-visible
    # copies since there is no P2P), sums them, and writes the result to its own
    # out buffer. After this, every device's out buffer holds the 2-rank sum.
    comptime for i in range(ngpus):

        @inline(.always)
        def outputs_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) {
            var out_tensors_capture
        } -> None:
            out_tensors_capture[i].store[width=_width, alignment=alignment](
                coords, rebind[SIMD[.float32, _width]](val)
            )

        _allreduce_naive_single[dtype=DType.float32, ngpus=ngpus](
            in_tensors, out_tensors[i].layout, outputs_lambda, 216, ctxs[i]
        )

    # --- Synchronize + read back + verify the sum on BOTH devices. -------------
    for i in range(ngpus):
        ctxs[i].synchronize()

    var expected = Float32(0)
    for i in range(ngpus):
        expected += Float32(i + 1)  # = 1 + 2 = 3
    for i in range(ngpus):
        ctxs[i].enqueue_copy(host_ptrs[i], out_dev[i])
        ctxs[i].synchronize()

    for i in range(ngpus):
        for j in range(length):
            assert_almost_equal(host_ptrs[i][j], expected)

    print(t"2-device host-staged allreduce: both B60s agree on the sum = {expected}")
    print("hal_test_allreduce_2dev PASSED")

    for i in range(ngpus):
        host_ptrs[i].unsafe_free()
