# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""`test_add_constant` run through the HAL `DeviceContext`.

`max.gpu.host.DeviceContext` is the NeoDriver-backed one) so a green run
proves device execution dispatches through the `M_driver_*` HAL plugin named
by `MODULAR_DRIVER_PLUGINS` (the Intel Level Zero plugin for the Arc Pro B60).

Mirrors the upstream `test` kernel but only deps on
the HAL `max` + `std` (no `internal_utils`/`linalg`/... needed). Unlike
`hal_test_id`, this exercises two capabilities the `test_id` path had not yet
exercised:

- `map_to_host()` — the host<->device buffer mapping (copy in on enter, copy
  out on exit). This is the supported path for our device-only / host-only
  memory model (device memory is not host-visible, so mapping is a DMA copy).
- scalar **by-value** kernel arguments (`constant: Float32`, `len_dev: Int32`)
  passed alongside the two buffer pointers — `test_id` only passed pointers.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def add_constant_fn(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    constant: Float32,
    len_dev: Int32,
):
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = input[tid] + constant


def run_add_constant(ctx: DeviceContext) raises:
    comptime length = 1024

    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Float32(i)

    var block_dim = 32
    comptime constant = Float32(33)

    comptime kernel = add_constant_fn
    ctx.enqueue_function[kernel](
        out_device,
        in_device,
        constant,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out_device.map_to_host() as out_host:
        for i in range(10):
            assert_equal(out_host[i], Float32(i) + constant)


def main() raises:
    with DeviceContext() as ctx:
        run_add_constant(ctx)
