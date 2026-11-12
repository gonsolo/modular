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
    _PrintfEmitPluginHookFnType,
)
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string, get_static_string
from std.ffi import CStringSpan, external_call
from std.io.file_descriptor import RawFileDescriptor
from std.io.io import _printf
from std.sys.info import CompilationTarget


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from an Intel Level Zero kernel through the
    OpenCL `printf` builtin.

    Device `printf` has a single output stream, so `file` is ignored.
    """
    _printf["%s"](cstr.ptr())


@inline(.always)
def _printf_emit[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor):
    """Formats `args` into `fmt` with the OpenCL `printf` builtin.

    Device `printf` has a single output stream, so `file` is ignored. The
    OpenCL/SPIR-V `printf` builtin is *variadic* (`const char *fmt, ...`):
    unlike CUDA's `vprintf(fmt, void*)`, it takes each argument directly as a
    variadic argument, not as a pointer to a packed arg block. So the format
    string is the single fixed argument and the *loaded* arg pack is handed
    over as the variadic arguments — the same convention `fcntl` in
    `sys/_libc.mojo` uses, where the loaded pack (not a pointer to it) makes
    the variadic args the values themselves rather than references.
    """
    _ = external_call["printf", Int32, num_fixed_args=1](
        get_static_string[fmt]().as_c_string_span(),
        args.get_loaded_kgen_pack(),
    )


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


struct LevelZeroPlugin(StdPlugin):
    """`PluginHooks` implementation for SPIR-V (Intel GPU) backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["level_zero"]()
    comptime HookFor[target: CompilationTarget] = LevelZeroPluginHooks[target]


struct LevelZeroPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _spirv_gpu_id_fn[name, dim]()

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime printf_emit_fn[fmt: StaticString, *types: AnyType]: OptionalReg[
        _PrintfEmitPluginHookFnType[*types]
    ] = _printf_emit[fmt, *types]
