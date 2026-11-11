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

#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Memory.h"

#include "llvm/ADT/Twine.h"

#include <vector>

using M::Target::LevelZero::zeErrorStatus;

ZeApi &M_driver_context::ze() { return device.ze(); }

DriverResult<M_driver_context_t> M_driver_context::create(M_driver_device &dev) {
  auto *ctx = new M_driver_context(dev);
  ze_context_desc_t desc = zeDesc<ze_context_desc_t>(ZE_STRUCTURE_TYPE_CONTEXT_DESC);
  if (ze_result_t r = dev.ze().zeContextCreate(dev.getDriver().zeDriver, &desc,
                                               &ctx->zeContext);
      r != ZE_RESULT_SUCCESS) {
    delete ctx;
    return tl::unexpected(
        zeErrorStatus(&dev.getDriver(), r, "zeContextCreate"));
  }
  return ctx;
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  ZeApi &ze = ctx->ze();
  if (ctx->syncList)
    ze.zeCommandListDestroy(ctx->syncList);
  if (ze_result_t r = ze.zeContextDestroy(ctx->zeContext);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&ctx->device.getDriver(), r, "zeContextDestroy");
  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<ze_command_list_handle_t> M_driver_context::getSyncList() {
  if (syncList)
    return syncList;
  ze_command_queue_desc_t desc = zeDesc<ze_command_queue_desc_t>(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC);
  desc.ordinal = device.computeOrdinal;
  desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  if (ze_result_t r = ze().zeCommandListCreateImmediate(
          zeContext, device.zeDevice, &desc, &syncList);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(zeErrorStatus(&device.getDriver(), r,
                                        "zeCommandListCreateImmediate"));
  return syncList;
}

M_driver_status_t M_driver_context::runSync(
    llvm::Twine const &what,
    llvm::function_ref<ze_result_t(ze_command_list_handle_t)> append) {
  std::lock_guard<std::mutex> lock(syncListMutex);
  DriverResult<ze_command_list_handle_t> list = getSyncList();
  if (!list)
    return list.error();
  if (ze_result_t r = append(*list); r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&device.getDriver(), r, what);
  if (ze_result_t r = ze().zeCommandListHostSynchronize(*list, UINT64_MAX);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&device.getDriver(), r, "zeCommandListHostSynchronize");
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_memory_t> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;
  ze_host_mem_alloc_desc_t desc = zeDesc<ze_host_mem_alloc_desc_t>(ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC);
  void *ptr = nullptr;
  if (ze_result_t r = ze().zeMemAllocHost(zeContext, &desc, size, 64, &ptr);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(r == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
                              ? M_DRIVER_STATUS_OUT_OF_MEMORY
                              : zeErrorStatus(&device.getDriver(), r,
                                              "zeMemAllocHost"));
  return new M_driver_memory{.ptr = ptr, .owning = true, .hostAccessible = true};
}

DriverResult<M_driver_memory_t> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;
  ze_device_mem_alloc_desc_t desc = zeDesc<ze_device_mem_alloc_desc_t>(ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC);
  desc.ordinal = 0;
  void *ptr = nullptr;
  if (ze_result_t r = ze().zeMemAllocDevice(zeContext, &desc, size, 256,
                                            device.zeDevice, &ptr);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(r == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
                              ? M_DRIVER_STATUS_OUT_OF_MEMORY
                              : zeErrorStatus(&device.getDriver(), r,
                                              "zeMemAllocDevice"));
  allocatedBytes += size;
  return new M_driver_memory{.ptr = ptr, .size = size};
}

DriverResult<M_driver_memory_t>
M_driver_context::wrapSync(uint64_t address, uint64_t size, bool owning) {
  if (address == 0)
    return tl::unexpected(M_DRIVER_STATUS_INVALID_ARG);
  return new M_driver_memory{.ptr = reinterpret_cast<void *>(address),
                             .owning = owning, .size = size};
}

M_driver_status_t M_driver_context::getCurrentDriverContext(
    M_driver_native_handle_t *driver_context) {
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *driver_context = reinterpret_cast<M_driver_native_handle_t>(zeContext);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::memoryInfo(uint64_t *free,
                                               uint64_t *total) {
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  uint32_t count = 0;
  ze().zeDeviceGetMemoryProperties(device.zeDevice, &count, nullptr);
  std::vector<ze_device_memory_properties_t> props(count, zeDesc<ze_device_memory_properties_t>(ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES));
  if (ze_result_t r =
          ze().zeDeviceGetMemoryProperties(device.zeDevice, &count, props.data());
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&device.getDriver(), r, "zeDeviceGetMemoryProperties");
  uint64_t bytes = 0;
  for (const ze_device_memory_properties_t &p : props)
    bytes += p.totalSize;
  *total = bytes;
  uint64_t used = allocatedBytes.load();
  *free = used < bytes ? bytes - used : 0;
  return M_DRIVER_STATUS_SUCCESS;
}

// zeMemFree is not ordered after queued work (cuMemFree synchronizes): free once the device is done with it.
ze_result_t M_driver_context::deferFree(void *ptr) {
  ze_memory_free_ext_desc_t desc =
      zeDesc<ze_memory_free_ext_desc_t>(ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC);
  desc.freePolicy = ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE;
  return ze().zeMemFreeExt(zeContext, &desc, ptr);
}

M_driver_status_t M_driver_context::freePinned(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  if (!memory->hostAccessible)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (ze_result_t r = deferFree(memory->ptr); r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&device.getDriver(), r, "zeMemFreeExt");
  delete memory;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  if (memory->owning) {
    if (ze_result_t r = deferFree(memory->ptr); r != ZE_RESULT_SUCCESS)
      return zeErrorStatus(&device.getDriver(), r, "zeMemFreeExt");
    allocatedBytes -= std::min<uint64_t>(memory->size, allocatedBytes.load());
  }
  delete memory;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  return runSync("copy to device", [&](ze_command_list_handle_t list) {
    return ze().zeCommandListAppendMemoryCopy(list, offsetPtr(dst, dstOffset),
                                              src, size, nullptr, 0, nullptr);
  });
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  return runSync("copy from device", [&](ze_command_list_handle_t list) {
    return ze().zeCommandListAppendMemoryCopy(
        list, dst, offsetPtr(src, srcOffset), size, nullptr, 0, nullptr);
  });
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size) {
  return runSync("copy intra device", [&](ze_command_list_handle_t list) {
    return ze().zeCommandListAppendMemoryCopy(list, offsetPtr(dst, dstOffset),
                                              offsetPtr(src, srcOffset), size,
                                              nullptr, 0, nullptr);
  });
}

M_driver_status_t M_driver_context::setMemorySync(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  uint64_t size, uint8_t value) {
  return fillSync(dst, dstOffset, size, value, 1);
}

M_driver_status_t M_driver_context::fillSync(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize != 1 && valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size % valueSize != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  return runSync("fill", [&](ze_command_list_handle_t list) {
    return ze().zeCommandListAppendMemoryFill(list, offsetPtr(dst, dstOffset),
                                              &value, valueSize, size, nullptr,
                                              0, nullptr);
  });
}
