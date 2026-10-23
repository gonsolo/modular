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

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std._plugin._trait import (
    TargetDeviceKind,
    PluginHooks,
    StdPlugin,
    _PrintEmitPluginHookFnType,
)
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.ffi import CStringSpan, external_call
from std.io.file_descriptor import RawFileDescriptor
from std.sys.info import CompilationTarget


def _drop_print[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    # TODO: OpenCL printf needs its format string in the constant space.
    pass


@inline(.always)
def _ocl_size[builtin: StaticString, dim: StaticString]() -> Int:
    # OpenCL work-item builtins; LLVM's SPIR-V backend lowers them to BuiltIns.
    comptime d = 0 if dim == "x" else (1 if dim == "y" else 2)
    return Int(external_call[builtin, UInt64](UInt32(d)))


def _spirv_gpu_id_fn[
    name: StaticString, dim: StaticString
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the OpenCL builtin read of `name` along `dim`, or `None` for
    a thread block cluster id, which SPIR-V does not have."""
    comptime if _is_cluster_id(name):
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_spirv_gpu_id[name, dim])
        )


@inline(.always)
def _spirv_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    comptime if name == "lane_id":
        return Int(external_call["_Z22get_sub_group_local_idv", UInt32]())
    elif name == "thread_idx":
        return _ocl_size["_Z12get_local_idj", dim]()
    elif name == "block_idx":
        return _ocl_size["_Z12get_group_idj", dim]()
    elif name == "block_dim":
        return _ocl_size["_Z14get_local_sizej", dim]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        return _ocl_size["_Z14get_num_groupsj", dim]()


struct SPIRVPlugin(StdPlugin):
    """`PluginHooks` implementation for SPIR-V (Intel GPU) backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SPIRVPluginHooks[target]


struct SPIRVPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _spirv_gpu_id_fn[name, dim]()

    comptime print_emit_fn: OptionalReg[_PrintEmitPluginHookFnType] = _drop_print
