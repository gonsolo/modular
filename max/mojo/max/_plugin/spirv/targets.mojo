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

"""Intel GPU targets, each carrying the SPIR-V stdlib and MAX plugins."""

from std.sys import CompilationTarget
from std._gpu.host.info import (
    AcceleratorArchitectureFamily,
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
    _K,
    _KB,
    _supported_archs_line,
)
from std._plugin.spirv import SPIRVPlugin

from . import SPIRVMaxPlugin

comptime SPIRV_PLUGINS = TypeList.of[SPIRVPlugin, SPIRVMaxPlugin]().values
"""The `opaque_plugin` payload of every Intel target."""


struct SPIRVTargets(TargetAcceleratorCollection):
    """Intel GPU targets that the SPIR-V plugins contribute."""

    comptime vendor_name: String = "Intel"

    comptime RAW_TARGETS = TypeList.of[
        Trait=TargetAcceleratorType,
        IntelBMG._with_cli_values[["bmg"]],
    ]().values

    @staticmethod
    def normalize_target_arch(target_arch0: StaticString) -> String:
        return target_arch0.replace("intel:", "")

    @staticmethod
    def supported_archs_message() -> String:
        return String("  ", _supported_archs_line[Self](), "\n\n")


comptime IntelXe2Family = AcceleratorArchitectureFamily(
    warp_size=16,
    threads_per_multiprocessor=8 * 8 * 16,
    shared_memory_per_multiprocessor=128 * _KB,
    max_registers_per_block=64 * _K,
    max_thread_block_size=_K,
)
"""Intel Xe2 GPU architecture family."""

comptime _intel_bmg_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-opencl", `,
        `stdlib_plugin = "spirv", `,
        `opaque_plugin = `,
        SPIRV_PLUGINS,
        `,`,
        `arch = "bmg", `,
        `features = "", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()
"""Target configuration for Intel Battlemage (Xe2) GPUs."""

comptime IntelBMG = TargetAccelerator[
    GPUInfo.from_family(
        family=IntelXe2Family,
        name="Arc B580",
        api="level_zero",
        arch_name="bmg",
        compute=2.0,  # Xe2
        version="xe2",
        sm_count=20,  # Xe2 cores
    ),
    _intel_bmg_target,
]()
"""Intel Arc B580 (Battlemage G21) configuration."""
