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

#include "Driver.h"

#include <memory>

using M::Target::LevelZero::zeErrorStatus;

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  DriverResult<ZeApi> api = ZeApi::load();
  if (!api)
    return tl::unexpected(api.error());
  auto driver = std::make_unique<M_driver_driver>(std::move(*api));
  ZeApi &ze = driver->ze();
  if (ze_result_t r = ze.zeInit(ZE_INIT_FLAG_GPU_ONLY); r != ZE_RESULT_SUCCESS) {
    driver->initStatus = zeErrorStatus(driver.get(), r, "zeInit");
    return driver.release();
  }
  uint32_t numDrivers = 0;
  ze.zeDriverGet(&numDrivers, nullptr);
  std::vector<ze_driver_handle_t> drivers(numDrivers);
  ze.zeDriverGet(&numDrivers, drivers.data());
  // The first driver with GPUs owns all devices this plugin exposes.
  for (ze_driver_handle_t zeDriver : drivers) {
    uint32_t numDevices = 0;
    ze.zeDeviceGet(zeDriver, &numDevices, nullptr);
    std::vector<ze_device_handle_t> devices(numDevices);
    ze.zeDeviceGet(zeDriver, &numDevices, devices.data());
    for (ze_device_handle_t device : devices) {
      ze_device_properties_t props = zeDesc<ze_device_properties_t>(ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES);
      if (ze.zeDeviceGetProperties(device, &props) == ZE_RESULT_SUCCESS &&
          props.type == ZE_DEVICE_TYPE_GPU)
        driver->devices.push_back(device);
    }
    if (!driver->devices.empty()) {
      driver->zeDriver = zeDriver;
      break;
    }
  }
  return driver.release();
}
