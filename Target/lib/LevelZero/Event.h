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

#ifndef TARGET_LIB_LEVELZERO_EVENT_H
#define TARGET_LIB_LEVELZERO_EVENT_H

#include "ZeApi.h"

#include "HAL/M_driver_event.h"

struct M_driver_context;

/// A host-visible event in its own one-slot pool.
struct M_driver_event {
  ze_event_pool_handle_t pool;
  ze_event_handle_t event;
  /// Unrecorded events count as complete, as in CUDA.
  bool recorded = false;

  M_driver_event(ze_event_pool_handle_t pool, ze_event_handle_t event)
      : pool(pool), event(event) {}

  [[nodiscard]] static DriverResult<M_driver_event_t>
  create(M_driver_context &ctx, M_driver_event_flags_t flags);
  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_event_t event);
  static M_driver_status_t synchronize(M_driver_context &ctx,
                                       M_driver_event_t event);
  static M_driver_status_t isReady(M_driver_context &ctx,
                                   M_driver_event_t event, bool *is_ready);
};

#endif // TARGET_LIB_LEVELZERO_EVENT_H
