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

#include "Device.h"
#include "Driver.h"

#include "HAL/M_driver_common.h"
#include "HAL/M_driver_device.h"

#include "llvm/ADT/Twine.h"

#include <cstring>
#include <dlpack/dlpack.h>
#include <vector>

using M::Target::LevelZero::zeErrorStatus;

DriverResult<M_driver_device_t> M_driver_device::create(M_driver_driver &drv,
                                                        int64_t deviceId) {
  if (deviceId < 0 || deviceId >= static_cast<int64_t>(drv.devices.size()))
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);
  ZeApi &ze = drv.ze();
  auto *device = new M_driver_device(drv, static_cast<int>(deviceId),
                                     drv.devices[deviceId]);
  if (ze_result_t r = ze.zeDeviceGetProperties(device->zeDevice, &device->props);
      r != ZE_RESULT_SUCCESS) {
    delete device;
    return tl::unexpected(zeErrorStatus(&drv, r, "zeDeviceGetProperties"));
  }
  ze.zeDeviceGetComputeProperties(device->zeDevice, &device->computeProps);
  uint32_t numGroups = 0;
  ze.zeDeviceGetCommandQueueGroupProperties(device->zeDevice, &numGroups,
                                            nullptr);
  std::vector<ze_command_queue_group_properties_t> groups(numGroups, zeDesc<ze_command_queue_group_properties_t>(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES));
  ze.zeDeviceGetCommandQueueGroupProperties(device->zeDevice, &numGroups,
                                            groups.data());
  for (uint32_t i = 0; i < numGroups; ++i)
    if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      device->computeOrdinal = i;
      break;
    }
  return device;
}

/// The compile target arch, as `--target-accelerator intel:<arch>` names it.
static const char *archName(uint32_t pciDeviceId) {
  switch (pciDeviceId) {
  case 0xE202: // Battlemage G21
  case 0xE20B: // Arc B580
  case 0xE20C: // Arc B570
  case 0xE20D:
  case 0xE210:
  case 0xE212:
  case 0xE215:
  case 0xE216:
    return "bmg";
  default:
    return "unknown";
  }
}

M_driver_status_t M_driver_device::property(llvm::StringRef name, void *value) {
  if (name.empty() || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  auto copyString = [&](llvm::StringRef str) {
    size_t n = std::min<size_t>(str.size(), M_DRIVER_DEVICE_PROPERTY_MAX_LEN - 1);
    char *out = static_cast<char *>(value);
    memcpy(out, str.data(), n);
    out[n] = '\0';
    return M_DRIVER_STATUS_SUCCESS;
  };
  if (name == M_DRIVER_DEVICE_PROPERTY_NAME)
    return copyString(props.name);
  if (name == M_DRIVER_DEVICE_PROPERTY_ARCH)
    return copyString(archName(props.deviceId));
  if (name == M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED) {
    *static_cast<int64_t *>(value) =
        (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) ? 1 : 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE) {
    *static_cast<uint64_t *>(value) = props.maxMemAllocSize;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLOneAPI;
    dl->device_id = static_cast<int32_t>(id());
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCPU;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device::canAccessPeer(const M_driver_device &peer,
                                                 bool *out_can_access) {
  if (out_can_access == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *out_can_access = false;
  return M_DRIVER_STATUS_SUCCESS;
}

// Callers pass CUDA `CUdevice_attribute` IDs; translate the ones they use.
M_driver_status_t M_driver_device::attribute(int32_t attribute,
                                             int32_t *value) {
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  switch (attribute) {
  case 6: // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
    *value = static_cast<int32_t>(std::min<uint32_t>(
        computeProps.maxGroupCountY, INT32_MAX));
    return M_DRIVER_STATUS_SUCCESS;
  case 16: // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    *value = static_cast<int32_t>(props.numSlices * props.numSubslicesPerSlice);
    return M_DRIVER_STATUS_SUCCESS;
  case 75: // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  case 76: // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  case 95: // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
    *value = 0;
    return M_DRIVER_STATUS_SUCCESS;
  default:
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  }
}
