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
//
// Object emission for SPIR-V (Intel GPU) targets: LLVM's SPIR-V backend writes
// a self-contained module that Level Zero loads with zeModuleCreate.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostBackend.h"
#include "Target/max_license/SPIRV/SPIRVTraits.h"

#include "Mojo/ToolCommon/CompilationOptions.h"

#include "llvm/IR/Module.h"

namespace M::KGEN {
namespace {

class SPIRVBackend final : public HostBackend {
public:
  const TargetTraits *traits() const override { return &SPIRVTraits::get(); }

  bool isCodegenInterprocedural() const override { return true; }

  bool isOffload() const override { return true; }

  // One SPIR-V module per compilation; there is no linker to combine splits.
  bool requiresSingleModuleSplit() const override { return true; }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return 3;
  }

  llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const override {
    return options.bonusInstThreshold(2);
  }

  void addSanitizers(llvm::ModulePassManager &,
                     const CompilationOptions &) const override {}

  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
  }

  // The arch names the device for the driver; LLVM's SPIR-V backend has no
  // processors.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const override {
    CompilationOptions adjusted = options;
    adjusted.targetCpu = "";
    return adjusted;
  }

  void finalizeModuleForTarget(llvm::Module &module, llvm::TargetMachine &,
                               llvm::StringRef) const override {
    for (llvm::Function &func : module)
      func.removeFnAttr("target-cpu");
  }

  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    WriteableBufferRef codeBuf = WriteableBuffer::get();
    if (ErrorOrSuccess err =
            ctx.runLlc(module, *codeBuf, /*createObjectFile=*/true))
      return Error(Twine(err.getError()) +
                   ", llc failed to codegen LLVM IR to SPIR-V");
    return BufferRef::take(codeBuf.release());
  }

  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override {
    if (objects.size() != 1)
      return Error("spirv backend expects a single object to archive");
    return objects[0];
  }

  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<SPIRVBackend> registerSPIRVBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
