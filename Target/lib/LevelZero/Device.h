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

#ifndef TARGET_LIB_LEVELZERO_DEVICE_H
#define TARGET_LIB_LEVELZERO_DEVICE_H

#include "Driver.h"
#include "Error.h"
#include "ZeApi.h"

#include "llvm/ADT/StringRef.h"

struct M_driver_device {
private:
  M_driver_driver &driver;
  int deviceId;

public:
  ze_device_handle_t zeDevice;
  ze_device_properties_t props = zeDesc<ze_device_properties_t>(ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES);
  ze_device_compute_properties_t computeProps = zeDesc<ze_device_compute_properties_t>(ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES);
  /// Ordinal of the command queue group that runs compute work.
  uint32_t computeOrdinal = 0;

  // Level Zero has no current device; kept for parity with the other plugins.
  DriverResult<void> deviceMethodPreamble() { return {}; }

  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &drv, int64_t deviceId);
  static M_driver_status_t destroy(M_driver_device_t device) {
    delete device;
    return M_DRIVER_STATUS_SUCCESS;
  }

  explicit M_driver_device(M_driver_driver &driver, int deviceId,
                           ze_device_handle_t zeDevice)
      : driver(driver), deviceId(deviceId), zeDevice(zeDevice) {}

  int id() const { return deviceId; }
  M_driver_driver &getDriver() { return driver; }
  ZeApi &ze() { return driver.ze(); }

  M_driver_status_t property(llvm::StringRef name, void *value);
  M_driver_status_t attribute(int32_t attribute, int32_t *value);
  M_driver_status_t canAccessPeer(const M_driver_device &peer,
                                  bool *out_can_access);
};

#endif // TARGET_LIB_LEVELZERO_DEVICE_H
