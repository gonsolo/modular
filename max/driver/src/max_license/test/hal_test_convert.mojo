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
"""`test_convert` (runtime part) run through the HAL `DeviceContext`.

the upstream `test` kernel.

The upstream test has two parts: `test_convert_asm` (a codegen-assertion that
checks for NVIDIA/AMD `cvt.*` / `v_cvt_*` asm strings, targeting `sm_80` and
`mi355x`) and `test_convert` (a runtime conversion). Only the **runtime** part
is portable to the Intel/SPIR-V target — the asm-string assertions are
platform-specific, so they are dropped here.

The runtime test builds a SIMD[Float32, 2] vector, casts it to
SIMD[BFloat16, 2], stores it, and verifies the round-trip value. This exercises
the SIMD `.cast[]` path (width-2) on the B60.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_equal


def convert_kernel[
    src_type: DType, dst_type: DType, size: Int
](dst_ptr: MutPointer[Scalar[dst_type], MutAnyOrigin]):
    comptime for i in range(0, size, 2):
        var src_vec = SIMD[src_type, 2](
            Scalar[src_type](i), Scalar[src_type](i + 1)
        )
        var dst_vec = src_vec.cast[dst_type]()
        dst_ptr.unsafe_store(i, dst_vec)


def test_convert[src_type: DType, dst_type: DType](ctx: DeviceContext) raises:
    """Test the conversion instruction (runtime result verification)."""

    comptime size = 4
    var device_buf = ctx.enqueue_create_buffer[dst_type](size)
    device_buf.enqueue_fill(0)

    comptime kernel = convert_kernel[src_type, dst_type, size]
    ctx.enqueue_function[kernel](device_buf, grid_dim=(1), block_dim=(1))
    with device_buf.map_to_host() as host_buf:
        for i in range(size):
            assert_equal(host_buf[i], Scalar[dst_type](i))


def main() raises:
    with DeviceContext() as ctx:
        # Only support 2xFP32 -> 2xBF16 conversion via the SIMD cast path.
        test_convert[.float32, .bfloat16](ctx)
        print("test_convert (runtime, 2xFP32 -> 2xBF16): PASSED")
