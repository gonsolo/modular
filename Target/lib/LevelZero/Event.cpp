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

#include "Event.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"

using M::Target::LevelZero::zeErrorStatus;

DriverResult<M_driver_event_t>
M_driver_event::create(M_driver_context &ctx, M_driver_event_flags_t flags) {
  M_driver_driver &driver = ctx.getDevice().getDriver();
  ZeApi &ze = ctx.ze();
  ze_event_pool_desc_t poolDesc = zeDesc<ze_event_pool_desc_t>(ZE_STRUCTURE_TYPE_EVENT_POOL_DESC);
  poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  ze_device_handle_t device = ctx.getDevice().zeDevice;
  ze_event_pool_handle_t pool = nullptr;
  if (ze_result_t r =
          ze.zeEventPoolCreate(ctx.zeContext, &poolDesc, 1, &device, &pool);
      r != ZE_RESULT_SUCCESS)
    return tl::unexpected(zeErrorStatus(&driver, r, "zeEventPoolCreate"));
  ze_event_desc_t desc = zeDesc<ze_event_desc_t>(ZE_STRUCTURE_TYPE_EVENT_DESC);
  desc.index = 0;
  desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ze_event_handle_t event = nullptr;
  if (ze_result_t r = ze.zeEventCreate(pool, &desc, &event);
      r != ZE_RESULT_SUCCESS) {
    ze.zeEventPoolDestroy(pool);
    return tl::unexpected(zeErrorStatus(&driver, r, "zeEventCreate"));
  }
  return new M_driver_event(pool, event);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  ZeApi &ze = ctx.ze();
  ze.zeEventDestroy(event->event);
  ze.zeEventPoolDestroy(event->pool);
  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  if (!event->recorded)
    return M_DRIVER_STATUS_SUCCESS;
  if (ze_result_t r = ctx.ze().zeEventHostSynchronize(event->event, UINT64_MAX);
      r != ZE_RESULT_SUCCESS)
    return zeErrorStatus(&ctx.getDevice().getDriver(), r,
                         "zeEventHostSynchronize");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (is_ready == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (!event->recorded) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  ze_result_t r = ctx.ze().zeEventQueryStatus(event->event);
  if (r == ZE_RESULT_SUCCESS || r == ZE_RESULT_NOT_READY) {
    *is_ready = r == ZE_RESULT_SUCCESS;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return zeErrorStatus(&ctx.getDevice().getDriver(), r, "zeEventQueryStatus");
}
