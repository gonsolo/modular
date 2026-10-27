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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstring>
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
// IGC cannot extract an aggregate member whole (its emitter copies into a register it never allocated): rebuild
// every aggregate-typed extractvalue from its leaves.
void scalarizeAggregateExtracts(llvm::Module &module) {
  llvm::SmallVector<llvm::ExtractValueInst *> worklist;
  for (llvm::Function &func : module)
    for (llvm::BasicBlock &block : func)
      for (llvm::Instruction &inst : block)
        if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(&inst))
          if (extract->getType()->isAggregateType())
            worklist.push_back(extract);
  for (llvm::ExtractValueInst *extract : worklist) {
    llvm::IRBuilder<> b(extract);
    llvm::Value *result = llvm::PoisonValue::get(extract->getType());
    llvm::SmallVector<unsigned> path;
    forEachLeaf(extract->getType(), path, [&](llvm::ArrayRef<unsigned> leaf) {
      result = b.CreateInsertValue(result, extractLeaf(b, extract, leaf), leaf);
    });
    extract->replaceAllUsesWith(result);
  }
  for (llvm::ExtractValueInst *extract : llvm::reverse(worklist))
    if (extract->use_empty())
      extract->eraseFromParent();
}

// LLVM's SPIR-V backend infers pointee types, and can give a pointer extracted from (or inserted into) a struct a
// different one than the member's: invalid SPIR-V that crashes IGC. Route such values through an OpBitcast.
void castCompositePointers(llvm::SmallVectorImpl<uint32_t> &words) {
  enum : uint32_t { TypeVector = 23, TypeArray = 28, TypeRuntimeArray = 29, TypeStruct = 30, TypePointer = 32,
                    Function = 54, CompositeExtract = 81, CompositeInsert = 82, Bitcast = 124 };
  if (words.size() < 5)
    return;
  llvm::DenseMap<uint32_t, llvm::SmallVector<uint32_t>> types;   // type id -> opcode, operands
  llvm::DenseMap<uint32_t, uint32_t> resultType;                   // value id -> its type id
  for (size_t i = 5; i < words.size();) {
    uint32_t op = words[i] & 0xffff, n = words[i] >> 16;
    if (n == 0 || i + n > words.size())
      return;
    if (op >= TypeVector && op <= TypePointer + 7 && n >= 2) {
      llvm::SmallVector<uint32_t> t = {op};
      t.append(words.begin() + i + 2, words.begin() + i + n);
      types[words[i + 1]] = t;
    } else if (n >= 3 && types.count(words[i + 1]) && !types.count(words[i + 2])) {
      resultType[words[i + 2]] = words[i + 1];
    }
    i += n;
  }
  auto isPointer = [&](uint32_t t) { auto it = types.find(t); return it != types.end() && it->second[0] == TypePointer; };
  auto member = [&](uint32_t t, llvm::ArrayRef<uint32_t> idx) -> uint32_t {
    for (uint32_t k : idx) {
      auto it = types.find(t);
      if (it == types.end())
        return 0;
      uint32_t op = it->second[0];
      if (op == TypeStruct && k + 1 < it->second.size())
        t = it->second[k + 1];
      else if (op == TypeArray || op == TypeRuntimeArray || op == TypeVector)
        t = it->second[1];
      else
        return 0;
    }
    return t;
  };
  uint32_t bound = words[3];
  llvm::SmallVector<uint32_t> out(words.begin(), words.begin() + 5);
  for (size_t i = 5; i < words.size();) {
    uint32_t op = words[i] & 0xffff, n = words[i] >> 16;
    llvm::ArrayRef<uint32_t> inst(words.data() + i, n);
    i += n;
    if (op == CompositeExtract && n >= 5) {
      uint32_t type = inst[1], id = inst[2], comp = inst[3];
      uint32_t m = member(resultType.lookup(comp), inst.drop_front(4));
      if (m && m != type && isPointer(m) && isPointer(type)) {
        uint32_t tmp = bound++;
        out.push_back(inst[0]); out.push_back(m); out.push_back(tmp); out.append(inst.begin() + 3, inst.end());
        out.append({(4u << 16) | Bitcast, type, id, tmp});
        continue;
      }
    } else if (op == CompositeInsert && n >= 6) {
      uint32_t type = inst[1], obj = inst[3];
      uint32_t m = member(type, inst.drop_front(5)), ot = resultType.lookup(obj);
      if (m && ot && m != ot && isPointer(m) && isPointer(ot)) {
        uint32_t tmp = bound++;
        out.append({(4u << 16) | Bitcast, m, tmp, obj});
        out.append(inst.begin(), inst.begin() + 3); out.push_back(tmp); out.append(inst.begin() + 4, inst.end());
        continue;
      }
    }
    out.append(inst.begin(), inst.end());
  }
  out[3] = bound;
  words.assign(out.begin(), out.end());
}

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

/// The leaf of `value` at `path`, looking through insertvalue chains.
llvm::Value *findLeaf(llvm::IRBuilder<> &b, llvm::Value *value, llvm::ArrayRef<unsigned> path) {
  while (auto *insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
    llvm::ArrayRef<unsigned> idx = insert->getIndices();
    if (idx.size() <= path.size() && llvm::equal(idx, path.take_front(idx.size()))) {
      value = insert->getInsertedValueOperand();
      path = path.drop_front(idx.size());
      if (path.empty())
        return value;
    } else if (idx.size() > path.size() || !llvm::equal(idx.take_front(path.size()), path)) {
      value = insert->getAggregateOperand();
    } else {
      break;
    }
  }
  return path.empty() ? value : extractLeaf(b, value, path);
}

bool isNarrowInt(llvm::Type *type) { return type->isIntegerTy(1) || type->isIntegerTy(8); }

bool hasNarrowIntLeaf(llvm::Type *type) {
  bool found = false;
  llvm::SmallVector<unsigned> path;
  forEachLeaf(type, path, [&](llvm::ArrayRef<unsigned> leaf) {
    found |= isNarrowInt(llvm::ExtractValueInst::getIndexedType(type, leaf));
  });
  return found;
}

bool hasNestedAggregate(llvm::Type *type) {
  if (!type->isAggregateType())
    return false;
  unsigned count = type->isStructTy() ? type->getStructNumElements() : type->getArrayNumElements();
  for (unsigned i = 0; i < count; ++i)
    if ((type->isStructTy() ? type->getStructElementType(i) : type->getArrayElementType())->isAggregateType())
      return true;
  return false;
}

// IGC garbles a nested struct returned from a function (the same SPIR-V is right when inlined): return a flat struct
// of its leaves and rebuild the nested value at each call. Bool and i8 leaves travel as i32: IGC makes a Bool leaf i8,
// and its PromoteInt8Type has no case for an i8 extractvalue, looping forever when one feeds i8 arithmetic.
void flattenNestedAggregateReturns(llvm::Module &module) {
  llvm::SmallVector<llvm::Function *> funcs;
  for (llvm::Function &func : module)
    if (!func.isDeclaration() && func.getCallingConv() != llvm::CallingConv::SPIR_KERNEL &&
        func.getReturnType()->isAggregateType() &&
        (hasNestedAggregate(func.getReturnType()) || hasNarrowIntLeaf(func.getReturnType())))
      funcs.push_back(&func);
  for (llvm::Function *old : funcs) {
    llvm::Type *nested = old->getReturnType();
    llvm::SmallVector<llvm::SmallVector<unsigned>> leaves;
    llvm::SmallVector<llvm::Type *> leafTypes;
    llvm::SmallVector<unsigned> path;
    forEachLeaf(nested, path, [&](llvm::ArrayRef<unsigned> leaf) {
      leaves.emplace_back(leaf);
      llvm::Type *t = llvm::ExtractValueInst::getIndexedType(nested, leaf);
      leafTypes.push_back(isNarrowInt(t) ? llvm::Type::getInt32Ty(module.getContext()) : t);
    });
    auto *flat = llvm::StructType::get(module.getContext(), leafTypes);
    auto *fty = llvm::FunctionType::get(flat, old->getFunctionType()->params(), old->isVarArg());
    auto *func = llvm::Function::Create(fty, old->getLinkage(), old->getAddressSpace(), "", &module);
    func->copyAttributesFrom(old);
    func->copyMetadata(old, 0);
    func->removeRetAttrs(llvm::AttributeFuncs::typeIncompatible(flat, {}));
    func->splice(func->begin(), old);
    for (auto [oldArg, arg] : llvm::zip(old->args(), func->args())) {
      arg.takeName(&oldArg);
      oldArg.replaceAllUsesWith(&arg);
    }
    for (llvm::BasicBlock &block : *func)
      if (auto *ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator())) {
        llvm::IRBuilder<> b(ret);
        llvm::Value *result = llvm::PoisonValue::get(flat);
        for (auto [i, leaf] : llvm::enumerate(leaves))
          result = b.CreateInsertValue(result, b.CreateZExtOrBitCast(findLeaf(b, ret->getReturnValue(), leaf), leafTypes[i]),
                                       {unsigned(i)});
        b.CreateRet(result);
        ret->eraseFromParent();
      }
    llvm::SmallVector<llvm::CallInst *> calls;
    for (llvm::User *user : old->users())
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(user); call && call->getCalledFunction() == old)
        calls.push_back(call);
    for (llvm::CallInst *call : calls) {
      llvm::IRBuilder<> b(call);
      llvm::SmallVector<llvm::Value *> args(call->args());
      llvm::CallInst *newCall = b.CreateCall(func, args);
      newCall->setCallingConv(call->getCallingConv());
      newCall->setAttributes(call->getAttributes().removeRetAttributes(
          module.getContext(), llvm::AttributeFuncs::typeIncompatible(flat, {})));
      llvm::SmallVector<llvm::User *> users(call->users());
      for (llvm::User *user : users)
        if (auto *ev = llvm::dyn_cast<llvm::ExtractValueInst>(user);
            ev && !ev->getType()->isAggregateType()) {
          auto it = llvm::find_if(leaves, [&](auto &leaf) { return llvm::equal(leaf, ev->getIndices()); });
          if (it != leaves.end()) {
            llvm::IRBuilder<> eb(ev);
            ev->replaceAllUsesWith(
                eb.CreateTruncOrBitCast(eb.CreateExtractValue(newCall, {unsigned(it - leaves.begin())}), ev->getType()));
            ev->eraseFromParent();
          }
        }
      if (!call->use_empty()) {
        llvm::Value *result = llvm::PoisonValue::get(nested);
        for (auto [i, leaf] : llvm::enumerate(leaves))
          result = b.CreateInsertValue(
              result,
              b.CreateTruncOrBitCast(b.CreateExtractValue(newCall, {unsigned(i)}),
                                     llvm::ExtractValueInst::getIndexedType(nested, leaf)),
              leaf);
        call->replaceAllUsesWith(result);
      }
      call->eraseFromParent();
    }
    func->takeName(old);
    if (old->use_empty())
      old->eraseFromParent();
  }
}

// IGC misreads nested members of a struct kernel argument passed by value: pass it as clang does for OpenCL C, a byval
// pointer, and read each extracted member through a GEP + load.
void byvalStructKernelArgs(llvm::Module &module) {
  llvm::SmallVector<llvm::Function *> kernels;
  for (llvm::Function &func : module)
    if (func.getCallingConv() == llvm::CallingConv::SPIR_KERNEL && !func.isDeclaration() &&
        llvm::any_of(func.args(), [](llvm::Argument &a) { return a.getType()->isAggregateType(); }))
      kernels.push_back(&func);
  const llvm::DataLayout &dl = module.getDataLayout();
  for (llvm::Function *old : kernels) {
    llvm::LLVMContext &ctx = module.getContext();
    llvm::SmallVector<llvm::Type *> params;
    for (llvm::Argument &a : old->args())
      params.push_back(a.getType()->isAggregateType() ? llvm::PointerType::get(ctx, 0) : a.getType());
    auto *fty = llvm::FunctionType::get(old->getReturnType(), params, old->isVarArg());
    auto *func = llvm::Function::Create(fty, old->getLinkage(), old->getAddressSpace(), "", &module);
    func->copyAttributesFrom(old);
    func->copyMetadata(old, 0);
    func->splice(func->begin(), old);
    llvm::IRBuilder<> b(&*func->getEntryBlock().getFirstInsertionPt());
    for (auto [oldArg, arg] : llvm::zip(old->args(), func->args())) {
      arg.takeName(&oldArg);
      if (!oldArg.getType()->isAggregateType()) {
        oldArg.replaceAllUsesWith(&arg);
        continue;
      }
      llvm::Type *type = oldArg.getType();
      llvm::Align align = dl.getABITypeAlign(type);
      func->removeParamAttrs(arg.getArgNo(), llvm::AttributeFuncs::typeIncompatible(arg.getType(), {}));
      func->addParamAttr(arg.getArgNo(), llvm::Attribute::getWithByValType(ctx, type));
      func->addParamAttr(arg.getArgNo(), llvm::Attribute::getWithAlignment(ctx, align));
      llvm::SmallVector<llvm::User *> users(oldArg.users());
      for (llvm::User *user : users) {
        auto *ev = llvm::dyn_cast<llvm::ExtractValueInst>(user);
        if (!ev)
          continue;
        llvm::IRBuilder<> eb(ev);
        llvm::SmallVector<llvm::Value *> idx{eb.getInt32(0)};
        for (unsigned i : ev->getIndices())
          idx.push_back(eb.getInt32(i));
        llvm::Value *gep = eb.CreateInBoundsGEP(type, &arg, idx);
        ev->replaceAllUsesWith(eb.CreateLoad(ev->getType(), gep));
        ev->eraseFromParent();
      }
      if (!oldArg.use_empty())
        oldArg.replaceAllUsesWith(b.CreateAlignedLoad(type, &arg, align));
    }
    func->takeName(old);
    old->eraseFromParent();
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
    byvalStructKernelArgs(module);
    flattenNestedAggregateReturns(module);
    scalarizeAggregatePhisAndSelects(module);
    scalarizeAggregateExtracts(module);
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
    {
      llvm::SmallVector<uint32_t> words(codeBuf->getBufferSize() / 4);
      std::memcpy(words.data(), codeBuf->getBufferStart(), words.size() * 4);
      castCompositePointers(words);
      codeBuf = WriteableBuffer::get();
      codeBuf->write(reinterpret_cast<const char *>(words.data()), words.size() * 4);
    }
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
