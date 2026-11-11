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

#ifndef TARGET_LIB_LEVELZERO_DRIVER_H
#define TARGET_LIB_LEVELZERO_DRIVER_H

#include "Error.h"
#include "ZeApi.h"

#include "Target/lib/Support/Utils.h"

#include <vector>

using M::Target::CustomStatus;

struct M_driver_driver {
private:
  ZeApi rawZeApi;

public:
  CustomStatus status;
  M_driver_status_t initStatus = M_DRIVER_STATUS_SUCCESS;
  ze_driver_handle_t zeDriver = nullptr;
  std::vector<ze_device_handle_t> devices;

  [[nodiscard]] static DriverResult<M_driver_driver_t> create();
  static M_driver_status_t destroy(M_driver_driver_t driver) {
    delete driver;
    return M_DRIVER_STATUS_SUCCESS;
  }

  M_driver_driver(ZeApi &&rawZeApi)
      : rawZeApi(std::move(rawZeApi)),
        status(M_DRIVER_NUM_CUSTOM_STATUS_CODES) {}
  ~M_driver_driver() { rawZeApi.unload(); }

  ZeApi &ze() { return rawZeApi; }
  DriverResult<int64_t> deviceCount() {
    return static_cast<int64_t>(devices.size());
  }
};

#endif // TARGET_LIB_LEVELZERO_DRIVER_H
