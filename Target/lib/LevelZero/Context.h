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

#ifndef TARGET_LIB_LEVELZERO_CONTEXT_H
#define TARGET_LIB_LEVELZERO_CONTEXT_H

#include "Error.h"
#include "Memory.h"
#include "ZeApi.h"

#include "HAL/M_driver_memory.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Twine.h"

#include <atomic>
#include <mutex>

struct M_driver_device;

struct M_driver_context {
private:
  M_driver_device &device;
  /// Synchronous immediate command list for the context-level copies and fills.
  ze_command_list_handle_t syncList = nullptr;
  std::mutex syncListMutex;
  /// Level Zero reports no free memory without sysman; track our own usage.
  std::atomic<uint64_t> allocatedBytes{0};

  DriverResult<ze_command_list_handle_t> getSyncList();
  M_driver_status_t runSync(llvm::Twine const &what,
                            llvm::function_ref<ze_result_t(
                                ze_command_list_handle_t)> append);

public:
  ze_context_handle_t zeContext = nullptr;

  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);
  static M_driver_status_t destroy(M_driver_context_t ctx);

  explicit M_driver_context(M_driver_device &device) : device(device) {}

  M_driver_device &getDevice() { return device; }
  ZeApi &ze();

  DriverResult<M_driver_memory_t> allocPinned(uint64_t size);
  DriverResult<M_driver_memory_t> allocSync(uint64_t size);
  DriverResult<M_driver_memory_t> wrapSync(uint64_t address, uint64_t size,
                                           bool owning);
  M_driver_status_t setCurrent() { return M_DRIVER_STATUS_SUCCESS; }
  M_driver_status_t
  getCurrentDriverContext(M_driver_native_handle_t *driver_context);
  M_driver_status_t
  setCurrentDriverContext(M_driver_native_handle_t driver_context) {
    return M_DRIVER_STATUS_SUCCESS;
  }
  M_driver_status_t memoryInfo(uint64_t *free, uint64_t *total);
  M_driver_status_t enablePeerAccess(M_driver_context &peer) {
    return M_DRIVER_STATUS_UNIMPLEMENTED;
  }
  M_driver_status_t freePinned(M_driver_memory_t memory);
  M_driver_status_t freeSync(M_driver_memory_t memory);
  M_driver_status_t copyToDeviceSync(M_driver_memory_t dst, uint64_t dstOffset,
                                     const void *src, uint64_t size);
  M_driver_status_t copyFromDeviceSync(void *dst, M_driver_memory_t src,
                                       uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyInterDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size,
                                        M_driver_context_t srcContext) {
    return M_DRIVER_STATUS_UNIMPLEMENTED;
  }
  M_driver_status_t setMemorySync(M_driver_memory_t dst, uint64_t dstOffset,
                                  uint64_t size, uint8_t value);
  M_driver_status_t fillSync(M_driver_memory_t dst, uint64_t dstOffset,
                             uint64_t size, uint64_t value, uint64_t valueSize);
};

/// Byte address `offset` bytes into `memory`.
inline void *offsetPtr(M_driver_memory_t memory, uint64_t offset) {
  return static_cast<char *>(memory->ptr) + offset;
}

#endif // TARGET_LIB_LEVELZERO_CONTEXT_H
