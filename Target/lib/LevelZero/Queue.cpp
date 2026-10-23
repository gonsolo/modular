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

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Event.h"

#include <vector>

using M::Target::LevelZero::zeErrorStatus;

DriverResult<M_driver_queue_t> M_driver_queue::create(M_driver_context &ctx) {
  M_driver_device &device = ctx.getDevice();
  ze_command_queue_desc_t desc = zeDesc<ze_command_queue_desc_t>(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC);
  desc.ordinal = device.computeOrdinal;
  desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  ze_command_list_handle_t list = nullptr;
  if (ze_result_t r = ctx.ze().zeCommandListCreateImmediate(
          ctx.zeContext, device.zeDevice, &desc, &list);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(zeErrorStatus(&device.getDriver(), r,
                                        "zeCommandListCreateImmediate"));
  return new M_driver_queue(ctx, list);
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (M_driver_status_t s = queue->synchronize(); s != M_DRIVER_STATUS_SUCCESS)
    return s;
  if (ze_result_t r = queue->context.ze().zeCommandListDestroy(queue->list);
      r != ZE_RESULT_SUCCESS)
    return queue->check(r, "zeCommandListDestroy");
  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::check(ze_result_t r, llvm::Twine const &what) {
  if (r == ZE_RESULT_SUCCESS)
    return M_DRIVER_STATUS_SUCCESS;
  return zeErrorStatus(&context.getDevice().getDriver(), r, what);
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  return check(context.ze().zeCommandListAppendMemoryCopy(
                   list, offsetPtr(dst, dstOffset), src, size, nullptr, 0,
                   nullptr),
               "zeCommandListAppendMemoryCopy to device");
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  return check(context.ze().zeCommandListAppendMemoryCopy(
                   list, dst, offsetPtr(src, srcOffset), size, nullptr, 0,
                   nullptr),
               "zeCommandListAppendMemoryCopy from device");
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  return check(context.ze().zeCommandListAppendMemoryCopy(
                   list, offsetPtr(dst, dstOffset), offsetPtr(src, srcOffset),
                   size, nullptr, 0, nullptr),
               "zeCommandListAppendMemoryCopy intra device");
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t dstOffset, uint64_t size,
                                            uint8_t value) {
  return fillMemory(dst, dstOffset, size, value, 1);
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize != 1 && valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size % valueSize != 0)
    return M_DRIVER_STATUS_INVALID_ARG;
  // The pattern is copied when the command is appended.
  return check(context.ze().zeCommandListAppendMemoryFill(
                   list, offsetPtr(dst, dstOffset), &value, valueSize, size,
                   nullptr, 0, nullptr),
               "zeCommandListAppendMemoryFill");
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  ZeApi &ze = context.ze();
  if (M_driver_status_t s =
          check(ze.zeCommandListAppendEventReset(list, event->event),
                "zeCommandListAppendEventReset");
      s != M_DRIVER_STATUS_SUCCESS)
    return s;
  event->recorded = true;
  return check(ze.zeCommandListAppendSignalEvent(list, event->event),
               "zeCommandListAppendSignalEvent");
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t num_events) {
  std::vector<ze_event_handle_t> handles;
  for (uint32_t i = 0; i < num_events; ++i)
    if (events[i] != nullptr && events[i]->recorded)
      handles.push_back(events[i]->event);
  if (handles.empty())
    return M_DRIVER_STATUS_SUCCESS;
  return check(context.ze().zeCommandListAppendWaitOnEvents(
                   list, static_cast<uint32_t>(handles.size()), handles.data()),
               "zeCommandListAppendWaitOnEvents");
}

// Level Zero has no host callbacks: drain the queue, then run the function.
M_driver_status_t M_driver_queue::launchHostFunc(void (*fn)(void *),
                                                 void *user_data) {
  if (fn == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (M_driver_status_t s = synchronize(); s != M_DRIVER_STATUS_SUCCESS)
    return s;
  fn(user_data);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  return check(context.ze().zeCommandListHostSynchronize(list, UINT64_MAX),
               "zeCommandListHostSynchronize");
}
