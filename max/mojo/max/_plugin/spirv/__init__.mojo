# ===----------------------------------------------------------------------=== #
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

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.ffi import external_call
from std.sys.info import CompilationTarget

from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from .._trait import MaxPlugin, MaxPluginHooks

# OpenCL CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE.
comptime _LOCAL_AND_GLOBAL_FENCE = UInt32(3)


@inline(.always)
def _barrier():
    external_call["_Z7barrierj", NoneType](_LOCAL_AND_GLOBAL_FENCE)


@inline(.always)
def _syncwarp(mask: Int):
    external_call["_Z17sub_group_barrierj", NoneType](_LOCAL_AND_GLOBAL_FENCE)


struct SPIRVMaxPlugin(MaxPlugin):
    """Provides MAX behavior for SPIR-V (Intel GPU) targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SPIRVMaxPluginHooks[target]


struct SPIRVMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    # The SPIR-V backend requests SIMD16 for every kernel.
    comptime warp_size = 16
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        var mask = external_call["_Z16sub_group_balloti", SIMD[DType.uint32, 4]](
            Int32(Int(val))
        )
        comptime if ret_type == DType.uint64:
            return (
                mask[0].cast[DType.uint64]() | (mask[1].cast[DType.uint64]() << 32)
            ).cast[ret_type]()
        return mask[0].cast[ret_type]()
