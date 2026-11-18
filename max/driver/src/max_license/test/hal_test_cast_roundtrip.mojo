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
"""`test_cast_roundtrip` run through the HAL `DeviceContext`.

the upstream `test` kernel but only deps on the
HAL `max` + `std`. Exercises real device numerics: a `Float32 -> BFloat16 ->
Float32` round-trip cast, and the `nan` / `inf` / `neg_inf` / `-0.0` edge
values, verified host-side after `map_to_host()`. (Does not read `grid_dim.x`
in-kernel, so it is unaffected by the `num_workgroups` POD-interpreter
limitation that blocks `test_grid_dim` / `test_launch_dims` /
`test_verify_buffers_gpu`.)
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true

from std.utils.numerics import inf, isnan, nan, neg_inf


def id(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = Float32(BFloat16(input[tid]))


@inline(.never)
def run_cast_roundtrip(ctx: DeviceContext) raises:
    print("== run_cast_roundtrip")

    comptime length = 1024

    var in_host = ctx.enqueue_create_host_buffer[.float32](length)

    for i in range(length):
        in_host[i] = Float32(i)

    in_host[4] = nan[.float32]()
    in_host[5] = inf[.float32]()
    in_host[6] = neg_inf[.float32]()
    in_host[7] = -0.0

    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    in_device.enqueue_copy_from(in_host)

    var block_dim = 32

    comptime kernel = id
    ctx.enqueue_function[kernel](
        in_device,
        out_device,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    var expected: List[Float32] = [
        0.0,
        1.0,
        2.0,
        3.0,
        nan[.float32](),
        inf[.float32](),
        neg_inf[.float32](),
        -0.0,
        8.0,
        9.0,
    ]
    with out_device.map_to_host() as out_host:
        for i in range(10):
            print("at index", i, "the value is", out_host[i])
            if isnan(expected[i]):
                assert_true(isnan(out_host[i]))
            else:
                assert_equal(expected[i], out_host[i])

    _ = in_device


def main() raises:
    with DeviceContext() as ctx:
        run_cast_roundtrip(ctx)
