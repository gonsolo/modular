//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Queue.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdlib>
#include <cstring>
#include <string>

using M::Target::LevelZero::zeErrorStatus;

static constexpr uint32_t kSPIRVMagic = 0x07230203;

DriverResult<M_driver_runtime_bundle_t>
M_driver_runtime_bundle::load(M_driver_context &ctx,
                              M_driver_static_bundle_t bundle,
                              M_driver_bundle_compilation_options_t opts) {
  if (bundle == nullptr || bundle->mapped_data.data == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);
  llvm::StringRef fileType(bundle->file_type, bundle->file_type_len);
  uint32_t magic = 0;
  if (bundle->mapped_data.size >= sizeof(magic))
    memcpy(&magic, bundle->mapped_data.data, sizeof(magic));
  if ((fileType != "spirv" && fileType != "object") || magic != kSPIRVMagic)
    return tl::unexpected(M_DRIVER_STATUS_INCOMPATIBLE);
  M_driver_device &device = ctx.getDevice();
  ze_module_desc_t desc = zeDesc<ze_module_desc_t>(ZE_STRUCTURE_TYPE_MODULE_DESC);
  desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  desc.inputSize = bundle->mapped_data.size;
  desc.pInputModule =
      reinterpret_cast<const uint8_t *>(bundle->mapped_data.data);
  // IGC picks 256 registers per kernel where 128 would spill: B580 VCM -29%, path tracing -6%.
  static const char *buildFlags = getenv("MODULAR_L0_BUILD_FLAGS");
  desc.pBuildFlags = buildFlags ? buildFlags : "-ze-intel-enable-auto-large-GRF-mode";
  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t r = ctx.ze().zeModuleCreate(ctx.zeContext, device.zeDevice, &desc,
                                          &module, &log);
  if (r != ZE_RESULT_SUCCESS) {
    std::string text;
    size_t size = 0;
    if (log && ctx.ze().zeModuleBuildLogGetString(log, &size, nullptr) ==
                   ZE_RESULT_SUCCESS) {
      text.resize(size);
      ctx.ze().zeModuleBuildLogGetString(log, &size, text.data());
    }
    if (log)
      ctx.ze().zeModuleBuildLogDestroy(log);
    return tl::unexpected(zeErrorStatus(&device.getDriver(), r,
                                        "zeModuleCreate: " + llvm::Twine(text)));
  }
  if (log)
    ctx.ze().zeModuleBuildLogDestroy(log);
  return new M_driver_runtime_bundle{.module = module};
}

M_driver_status_t
M_driver_runtime_bundle::unload(M_driver_context &ctx,
                                M_driver_runtime_bundle_t bundle) {
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (ze_result_t r = ctx.ze().zeModuleDestroy(bundle->module);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&ctx.getDevice().getDriver(), r, "zeModuleDestroy");
  delete bundle;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_function_t>
M_driver_function::load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
                        const char *function_name, uint64_t function_name_len) {
  if (bundle == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_BAD_BUNDLE);
  std::string name(function_name, function_name_len);
  ze_kernel_desc_t desc = zeDesc<ze_kernel_desc_t>(ZE_STRUCTURE_TYPE_KERNEL_DESC);
  desc.pKernelName = name.c_str();
  ze_kernel_handle_t kernel = nullptr;
  if (ze_result_t r = ctx.ze().zeKernelCreate(bundle->module, &desc, &kernel);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(zeErrorStatus(&ctx.getDevice().getDriver(), r,
                                        "zeKernelCreate " + llvm::Twine(name)));
  // Kernels reach buffers through pointers inside structs; without this NEO binds only buffers passed as arguments.
  if (ze_result_t r = ctx.ze().zeKernelSetIndirectAccess(
          kernel, ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE | ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
                      ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED);
      r != ZE_RESULT_SUCCESS) {
    ctx.ze().zeKernelDestroy(kernel);
    return tl::unexpected(zeErrorStatus(&ctx.getDevice().getDriver(), r,
                                        "zeKernelSetIndirectAccess " + llvm::Twine(name)));
  }
  auto *func = new M_driver_function(ctx, kernel);
  func->name = name;
  return func;
}

M_driver_status_t M_driver_function::unload(M_driver_function_t func) {
  if (func == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  func->context.ze().zeKernelDestroy(func->kernel);
  delete func;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_function::launch(M_driver_queue &queue,
                          const M_driver_queue_execute_config_gpu_t &gpu,
                          void **args, uint64_t *arg_sizes, uint32_t num_args) {
  // Dynamic shared memory would need a local-pointer kernel argument.
  if (gpu.num_attributes > 0 || gpu.shared_mem_bytes > 0)
    return M_DRIVER_STATUS_UNIMPLEMENTED;
  ZeApi &ze = context.ze();
  M_driver_driver &driver = context.getDevice().getDriver();
  std::lock_guard<std::mutex> lock(launchMutex);
  if (ze_result_t r =
          ze.zeKernelSetGroupSize(kernel, gpu.block.x, gpu.block.y, gpu.block.z);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&driver, r,
                         llvm::Twine("zeKernelSetGroupSize ") +
                             llvm::Twine(gpu.block.x) +
                             "x" + llvm::Twine(gpu.block.y) + "x" +
                             llvm::Twine(gpu.block.z));
  for (uint32_t i = 0; i < num_args; ++i)
    if (ze_result_t r = ze.zeKernelSetArgumentValue(kernel, i, arg_sizes[i],
                                                    args[i]);
        r != ZE_RESULT_SUCCESS)
      return zeErrorStatus(&driver, r,
                           "zeKernelSetArgumentValue " + llvm::Twine(i) +
                               " size " + llvm::Twine(arg_sizes[i]));
  ze_group_count_t groups = {gpu.grid.x, gpu.grid.y, gpu.grid.z};
  if (ze_result_t r = ze.zeCommandListAppendLaunchKernel(
          queue.list, kernel, &groups, nullptr, 0, nullptr);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&driver, r, "zeCommandListAppendLaunchKernel " + llvm::Twine(name));
  return M_DRIVER_STATUS_SUCCESS;
}
