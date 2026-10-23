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

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
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
                             "+SPV_INTEL_subgroups";
    if (auto extra = llvm::sys::Process::GetEnv("MOJO_SPIRV_EXT"))
      extensions += "," + *extra;
    it->second->addOccurrence(0, "spirv-ext", extensions);
  });
}

/// Calls `fn` with the index path of every scalar leaf of `type`.
void forEachLeaf(llvm::Type *type, llvm::SmallVectorImpl<unsigned> &path,
                 llvm::function_ref<void(llvm::ArrayRef<unsigned>)> fn) {
  if (!type->isAggregateType())
    return fn(path);
  unsigned count = type->isStructTy() ? type->getStructNumElements()
                                      : type->getArrayNumElements();
  for (unsigned i = 0; i < count; ++i) {
    path.push_back(i);
    forEachLeaf(type->isStructTy() ? type->getStructElementType(i)
                                   : type->getArrayElementType(),
                path, fn);
    path.pop_back();
  }
}

/// Extracts the leaf at `path`, looking through aggregate-valued extractvalues.
llvm::Value *extractLeaf(llvm::IRBuilder<> &b, llvm::Value *value,
                         llvm::ArrayRef<unsigned> path) {
  llvm::SmallVector<unsigned> indices(path);
  while (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(value)) {
    indices.insert(indices.begin(), extract->idx_begin(), extract->idx_end());
    value = extract->getAggregateOperand();
  }
  return b.CreateExtractValue(value, indices);
}

/// LLVM's SPIR-V backend replaces aggregate values by i32 placeholders and
/// miscompiles aggregate phis and selects; rewrite them field by field.
void scalarizeAggregatePhisAndSelects(llvm::Module &module) {
  llvm::SmallVector<llvm::Instruction *> worklist;
  for (llvm::Function &func : module)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block)
        if ((llvm::isa<llvm::PHINode>(inst) || llvm::isa<llvm::SelectInst>(inst)) &&
            inst.getType()->isAggregateType())
          worklist.push_back(&inst);
  for (llvm::Instruction *inst : worklist) {
    llvm::Type *type = inst->getType();
    llvm::IRBuilder<> b(inst);
    if (llvm::isa<llvm::PHINode>(inst))
      b.SetInsertPoint(inst->getParent()->getFirstNonPHIIt());
    llvm::Value *result = llvm::PoisonValue::get(type);
    llvm::SmallVector<unsigned> path;
    forEachLeaf(type, path, [&](llvm::ArrayRef<unsigned> leaf) {
      llvm::Value *scalar;
      if (auto *phi = llvm::dyn_cast<llvm::PHINode>(inst)) {
        llvm::IRBuilder<> pb(phi);
        auto *leafPhi = pb.CreatePHI(llvm::ExtractValueInst::getIndexedType(type, leaf),
                                     phi->getNumIncomingValues());
        for (unsigned i = 0, e = phi->getNumIncomingValues(); i < e; ++i) {
          llvm::IRBuilder<> eb(phi->getIncomingBlock(i)->getTerminator());
          leafPhi->addIncoming(extractLeaf(eb, phi->getIncomingValue(i), leaf),
                               phi->getIncomingBlock(i));
        }
        scalar = leafPhi;
      } else {
        auto *select = llvm::cast<llvm::SelectInst>(inst);
        scalar = b.CreateSelect(select->getCondition(),
                                extractLeaf(b, select->getTrueValue(), leaf),
                                extractLeaf(b, select->getFalseValue(), leaf));
      }
      result = b.CreateInsertValue(result, scalar, leaf);
    });
    inst->replaceAllUsesWith(result);
    inst->eraseFromParent();
  }
}

/// Stores `value` field by field, for aggregate stores the SPIR-V backend
/// rejects (anything but an extractvalue).
void storeFields(llvm::IRBuilder<> &b, llvm::Type *rootType, llvm::Value *root,
                 llvm::Value *ptr, llvm::SmallVectorImpl<unsigned> &path,
                 llvm::Type *type, llvm::Align align) {
  if (!type->isAggregateType()) {
    llvm::SmallVector<llvm::Value *> indices = {b.getInt32(0)};
    for (unsigned i : path)
      indices.push_back(b.getInt32(i));
    llvm::Value *field = b.CreateExtractValue(root, path);
    b.CreateAlignedStore(field, b.CreateInBoundsGEP(rootType, ptr, indices),
                         llvm::Align(1));
    return;
  }
  unsigned count = type->isStructTy() ? type->getStructNumElements()
                                      : type->getArrayNumElements();
  for (unsigned i = 0; i < count; ++i) {
    path.push_back(i);
    llvm::Type *elem = type->isStructTy() ? type->getStructElementType(i)
                                          : type->getArrayElementType();
    storeFields(b, rootType, root, ptr, path, elem, align);
    path.pop_back();
  }
}

void splitAggregateStores(llvm::Module &module) {
  llvm::SmallVector<llvm::StoreInst *> stores;
  for (llvm::Function &func : module)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block)
        if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
          if (store->getValueOperand()->getType()->isAggregateType() &&
              !llvm::isa<llvm::ExtractValueInst>(store->getValueOperand()))
            stores.push_back(store);
  for (llvm::StoreInst *store : stores) {
    llvm::IRBuilder<> b(store);
    llvm::Value *value = store->getValueOperand();
    llvm::SmallVector<unsigned> path;
    storeFields(b, value->getType(), value, store->getPointerOperand(), path,
                value->getType(), store->getAlign());
    store->eraseFromParent();
  }
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
    scalarizeAggregatePhisAndSelects(module);
    splitAggregateStores(module);
    if (auto path = llvm::sys::Process::GetEnv("MOJO_SPIRV_DUMP_IR")) {
      std::error_code ec;
      llvm::raw_fd_ostream os(*path, ec);
      if (!ec)
        module.print(os, nullptr);
    }
    WriteableBufferRef codeBuf = WriteableBuffer::get();
    if (ErrorOrSuccess err =
            ctx.runLlc(module, *codeBuf, /*createObjectFile=*/true))
      return Error(Twine(err.getError()) +
                   ", llc failed to codegen LLVM IR to SPIR-V");
    if (auto dir = llvm::sys::Process::GetEnv("MOJO_SPIRV_DUMP_DIR")) {
      static std::atomic<unsigned> counter{0};
      std::error_code ec;
      llvm::raw_fd_ostream os(*dir + "/" + std::to_string(counter++) + ".spv",
                              ec);
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
