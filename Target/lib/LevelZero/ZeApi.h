//===----------------------------------------------------------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Wraps the Level Zero C API as loaded through `dlopen`, so the plugin has no
// link-time dependency on libze_loader.so. Add new calls to ZE_FORALL_FUNCTIONS.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_LEVELZERO_ZEAPI_H
#define TARGET_LIB_LEVELZERO_ZEAPI_H

#include "Error.h"

#include "level_zero/ze_api.h"

#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
#include <stdio.h>
#include <utility>

#define ZE_FORALL_FUNCTIONS(ZE_FN)                                             \
  ZE_FN(zeInit)                                                                \
  ZE_FN(zeDriverGet)                                                           \
  ZE_FN(zeDeviceGet)                                                           \
  ZE_FN(zeDeviceGetProperties)                                                 \
  ZE_FN(zeDeviceGetComputeProperties)                                          \
  ZE_FN(zeDeviceGetMemoryProperties)                                           \
  ZE_FN(zeDeviceGetCommandQueueGroupProperties)                                \
  ZE_FN(zeContextCreate)                                                       \
  ZE_FN(zeContextDestroy)                                                      \
  ZE_FN(zeMemAllocDevice)                                                      \
  ZE_FN(zeMemAllocHost)                                                        \
  ZE_FN(zeMemFree)                                                             \
  ZE_FN(zeMemFreeExt)                                                          \
  ZE_FN(zeMemGetAllocProperties)                                               \
  ZE_FN(zeCommandListCreateImmediate)                                          \
  ZE_FN(zeCommandListDestroy)                                                  \
  ZE_FN(zeCommandListHostSynchronize)                                          \
  ZE_FN(zeCommandListAppendMemoryCopy)                                         \
  ZE_FN(zeCommandListAppendMemoryFill)                                         \
  ZE_FN(zeCommandListAppendLaunchKernel)                                       \
  ZE_FN(zeCommandListAppendSignalEvent)                                        \
  ZE_FN(zeCommandListAppendWaitOnEvents)                                       \
  ZE_FN(zeCommandListAppendEventReset)                                         \
  ZE_FN(zeEventPoolCreate)                                                     \
  ZE_FN(zeEventPoolDestroy)                                                    \
  ZE_FN(zeEventCreate)                                                         \
  ZE_FN(zeEventDestroy)                                                        \
  ZE_FN(zeEventHostSynchronize)                                                \
  ZE_FN(zeEventQueryStatus)                                                    \
  ZE_FN(zeModuleCreate)                                                        \
  ZE_FN(zeModuleDestroy)                                                       \
  ZE_FN(zeModuleBuildLogGetString)                                             \
  ZE_FN(zeModuleBuildLogDestroy)                                               \
  ZE_FN(zeKernelCreate)                                                        \
  ZE_FN(zeKernelDestroy)                                                       \
  ZE_FN(zeKernelSetGroupSize)                                                  \
  ZE_FN(zeKernelSetArgumentValue)

/// A zero-initialized Level Zero descriptor or properties struct of type `stype`.
template <typename T> T zeDesc(ze_structure_type_t stype) {
  T desc{};
  desc.stype = stype;
  return desc;
}

struct ZeApi {
  void *lib = nullptr;

#define DECLARE_FN_PTR(name) decltype(&::name) name = nullptr;
  ZE_FORALL_FUNCTIONS(DECLARE_FN_PTR)
#undef DECLARE_FN_PTR

  ZeApi(const ZeApi &) = delete;
  ZeApi &operator=(const ZeApi &) = delete;
  ZeApi(ZeApi &&other) noexcept { *this = std::move(other); }
  ZeApi &operator=(ZeApi &&other) noexcept {
    lib = other.lib;
    other.lib = nullptr;
#define MOVE_FN_PTR(name) name = other.name;
    ZE_FORALL_FUNCTIONS(MOVE_FN_PTR)
#undef MOVE_FN_PTR
    return *this;
  }
  ZeApi() = default;

  /// dlopens the Level Zero loader and resolves all symbols.
  static DriverResult<ZeApi> load() {
    void *lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib)
      lib = dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
      fprintf(stderr, "LevelZeroPlugin: dlopen(libze_loader.so) failed: %s\n",
              dlerror());
      return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
    }
    ZeApi api;
    api.lib = lib;
#define LOAD_SYM(name)                                                         \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(lib, #name));          \
  if (!api.name) {                                                             \
    fprintf(stderr, "LevelZeroPlugin: dlsym(%s) failed: %s\n", #name,          \
            dlerror());                                                        \
    api.unload();                                                              \
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);                      \
  }
    ZE_FORALL_FUNCTIONS(LOAD_SYM)
#undef LOAD_SYM
    return api;
  }

  void unload() {
    if (lib) {
      dlclose(lib);
      lib = nullptr;
    }
  }
};

struct M_driver_driver;

namespace M::Target::LevelZero {
/// Records `result` with `context` as an ephemeral status message.
M_driver_status_t zeErrorStatus(M_driver_driver *driver, ze_result_t result,
                                llvm::Twine const &context);
} // namespace M::Target::LevelZero

#endif // TARGET_LIB_LEVELZERO_ZEAPI_H
