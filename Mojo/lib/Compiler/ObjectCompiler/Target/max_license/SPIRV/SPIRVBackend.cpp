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

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <mutex>

namespace M::KGEN {
namespace {

/// Enables the SPIR-V extensions Intel's driver supports; LLVM's SPIR-V backend
/// only takes them from its `-spirv-ext` option.
void enableIntelExtensions() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto &options = llvm::cl::getRegisteredOptions();
    auto it = options.find("spirv-ext");
    if (it == options.end())
      return;
    std::string extensions = "+SPV_EXT_shader_atomic_float_add,"
                             "+SPV_EXT_shader_atomic_float_min_max,"
                             "+SPV_INTEL_subgroups,"
                             "+SPV_INTEL_masked_gather_scatter,"
                             "+SPV_EXT_relaxed_printf_string_address_space";
    if (auto extra = llvm::sys::Process::GetEnv("MOJO_SPIRV_EXT"))
      extensions += "," + *extra;
    it->second->addOccurrence(0, "spirv-ext", extensions);
  });
}

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

  // Kernels run SIMD16, the warp size the MAX plugin reports.
  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
    llvm::LLVMContext &ctx = kernelEntry->getContext();
    kernelEntry->setMetadata(
        "intel_reqd_sub_group_size",
        llvm::MDNode::get(ctx, llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(ctx), 16))));
  }

  // The arch names the device for the driver; LLVM's SPIR-V backend has no
  // processors.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const override {
    enableIntelExtensions();
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
    if (auto path = llvm::sys::Process::GetEnv("MOJO_SPIRV_DUMP_IR")) {
      std::error_code ec;
      llvm::raw_fd_ostream os(*path, ec);
      if (!ec)
        module.print(os, nullptr);
    }
    // Each module's IR as N.ll before codegen, its SPIR-V as N.spv after: a
    // module that crashed llc is the .ll without a .spv.
    static std::atomic<unsigned> counter{0};
    auto dir = llvm::sys::Process::GetEnv("MOJO_SPIRV_DUMP_DIR");
    std::string stem = dir ? *dir + "/" + std::to_string(counter++) : "";
    if (dir) {
      std::error_code ec;
      llvm::raw_fd_ostream os(stem + ".ll", ec);
      if (!ec)
        module.print(os, nullptr);
    }
    WriteableBufferRef codeBuf = WriteableBuffer::get();
    if (ErrorOrSuccess err =
            ctx.runLlc(module, *codeBuf, /*createObjectFile=*/true))
      return Error(Twine(err.getError()) +
                   ", llc failed to codegen LLVM IR to SPIR-V");
    if (dir) {
      std::error_code ec;
      llvm::raw_fd_ostream os(stem + ".spv", ec);
      if (!ec)
        os.write(codeBuf->getBufferStart(), codeBuf->getBufferSize());
    }
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
