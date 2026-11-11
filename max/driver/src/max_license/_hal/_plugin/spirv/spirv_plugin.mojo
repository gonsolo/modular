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

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget

from .._trait import HalPlugin, HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config.gpu_execution_config import GPUExecutionConfiguration
from ...extensions.traits import DefaultHALExtensions


struct SPIRVHalPlugin(HalPlugin):
    """HAL plugin for SPIR-V (Intel GPU) targets, run by the Level Zero
    driver plugin."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["spirv"]()
    comptime HookFor[target: CompilationTarget] = SPIRVHalPluginHooks[target]


struct SPIRVHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration
    comptime extensions = DefaultHALExtensions
