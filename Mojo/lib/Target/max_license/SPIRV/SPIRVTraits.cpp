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

#include "Target/max_license/SPIRV/SPIRVTraits.h"

namespace M::KGEN {

const SPIRVTraits &SPIRVTraits::get() {
  static const SPIRVTraits instance;
  return instance;
}

// Mojo's LOCAL; remapped to SPIR-V Function storage after lowering.
std::optional<unsigned> SPIRVTraits::requiredStackAllocationAddressSpace() const {
  return 5;
}

unsigned mojoToSPIRVAddressSpace(unsigned addressSpace) {
  switch (addressSpace) {
  case 0: // generic
    return 4;
  case 4: // constant
    return 2;
  case 5: // local (private)
    return 0;
  default: // global (1) and shared (3) agree
    return addressSpace;
  }
}

static constexpr TargetTraits::AcceleratorArch kSPIRVAcceleratorArchs[] = {
    {"bmg", "Xe2 (Arc B580/B570)"},
    {"intel:<arch>", "(alias) e.g., intel:bmg"},
};

llvm::ArrayRef<TargetTraits::AcceleratorArch>
SPIRVTraits::supportedAcceleratorArchs() const {
  return kSPIRVAcceleratorArchs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<SPIRVTraits> registerSPIRVTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
