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
// MLIR-lowering policy for SPIR-V (Intel GPU) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"
#include "Target/max_license/SPIRV/SPIRVTraits.h"

#include "KGENToLLVM/Target/max_license/GPUCommon.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace M::KGEN {
namespace {

constexpr unsigned kFunctionAS = 0;
constexpr unsigned kCrossWorkgroupAS = 1;
constexpr unsigned kGenericAS = 4;

/// Renumbers Mojo's (NVVM) address spaces to SPIR-V's and gives kernels the
/// OpenCL ABI: pointer arguments in CrossWorkgroup (byval ones in Function)
/// and module globals in CrossWorkgroup instead of the generic space.
struct SPIRVAddressSpacePass
    : public PassWrapper<SPIRVAddressSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SPIRVAddressSpacePass)

  StringRef getArgument() const final { return "kgen-spirv-address-spaces"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::DenseMap<StringAttr, unsigned> globalSpaces;
    module.walk([&](LLVM::GlobalOp global) {
      unsigned as = global.getAddrSpace();
      globalSpaces[global.getSymNameAttr()] = as;
      global.setAddrSpace(as == 0 ? kCrossWorkgroupAS
                                  : mojoToSPIRVAddressSpace(as));
    });

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](LLVM::LLVMPointerType type) -> Type {
      return LLVM::LLVMPointerType::get(
          ctx, mojoToSPIRVAddressSpace(type.getAddressSpace()));
    });
    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    auto generic = LLVM::LLVMPointerType::get(ctx, kGenericAS);
    module.walk([&](LLVM::AddressOfOp addr) {
      auto it = globalSpaces.find(addr.getGlobalNameAttr().getAttr());
      if (it == globalSpaces.end() || it->second != 0)
        return;
      addr.getResult().setType(
          LLVM::LLVMPointerType::get(ctx, kCrossWorkgroupAS));
      OpBuilder b(addr->getBlock(), std::next(addr->getIterator()));
      auto cast = LLVM::AddrSpaceCastOp::create(b, addr.getLoc(), generic,
                                                addr.getResult());
      addr.getResult().replaceAllUsesExcept(cast, cast);
    });

    module.walk([&](LLVM::LLVMFuncOp func) {
      if (func.getCConv() != LLVM::cconv::CConv::SPIR_KERNEL || func.isExternal())
        return;
      Block &entry = func.getBody().front();
      OpBuilder b(&entry, entry.begin());
      SmallVector<Type> params(func.getFunctionType().getParams());
      for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
        auto ptr = dyn_cast<LLVM::LLVMPointerType>(arg.getType());
        if (!ptr || ptr.getAddressSpace() != kGenericAS)
          continue;
        bool byVal = static_cast<bool>(
            func.getArgAttr(i, LLVM::LLVMDialect::getByValAttrName()));
        auto argType = LLVM::LLVMPointerType::get(
            ctx, byVal ? kFunctionAS : kCrossWorkgroupAS);
        arg.setType(argType);
        params[i] = argType;
        auto cast = LLVM::AddrSpaceCastOp::create(b, func.getLoc(), generic, arg);
        arg.replaceAllUsesExcept(cast, cast);
      }
      func.setFunctionType(LLVM::LLVMFunctionType::get(
          func.getFunctionType().getReturnType(), params,
          func.getFunctionType().isVarArg()));
    });
  }
};

class SPIRVLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &SPIRVTraits::get(); }

  mlir::Type lowerKernelArgToMemory(mlir::Type type) const override {
    return lowerAggregateArgToMemory(type);
  }

  bool needsVerification() const override { return true; }

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return verifyNoGPUHeapAllocation(op);
  }

  void addPostLowerToLLVMPasses(mlir::OpPassManager &pm) const override {
    pm.addPass(std::make_unique<SPIRVAddressSpacePass>());
  }

  void markExportedKernel(mlir::Operation *func) const override {
    cast<LLVM::LLVMFuncOp>(func).setCConv(LLVM::cconv::CConv::SPIR_KERNEL);
  }

  bool isExportedKernel(mlir::Operation *func) const override {
    auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(func);
    return llvmFunc && llvmFunc.getCConv() == LLVM::cconv::CConv::SPIR_KERNEL;
  }

  llvm::StringRef getKernelByValArgAttrName() const override {
    return mlir::LLVM::LLVMDialect::getByValAttrName();
  }

  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<SPIRVLowering> registerSPIRVLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
