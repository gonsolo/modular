//===----------------------------------------------------------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_SPIRV_SPIRVTRAITS_H
#define KGEN_TARGET_SPIRV_SPIRVTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

/// SPIR-V kernels for OpenCL-style drivers (Intel Level Zero).
struct SPIRVTraits final : TargetTraits {
  llvm::StringRef name() const override { return "spirv"; }
  bool isGPU() const override { return true; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.isSPIRV();
  }
  llvm::StringRef getAsmExtension() const override { return ".spvasm"; }
  llvm::StringRef getLLVMExtension() const override { return ".spirv.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".spv"; }
  llvm::StringRef getBitcodeExtension() const override { return ".bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Intel (SPIR-V/Level Zero):";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;
  std::optional<unsigned> requiredStackAllocationAddressSpace() const override;

  bool isBaseTarget() const override { return false; }

  static const SPIRVTraits &get();
};

/// Mojo's address spaces follow NVVM; SPIR-V numbers them differently.
unsigned mojoToSPIRVAddressSpace(unsigned addressSpace);

} // namespace M::KGEN

#endif // KGEN_TARGET_SPIRV_SPIRVTRAITS_H
