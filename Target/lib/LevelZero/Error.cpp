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

#include "Error.h"
#include "Driver.h"

#include "llvm/ADT/Twine.h"

namespace M::Target::LevelZero {

M_driver_status_t zeErrorStatus(M_driver_driver *driver, ze_result_t result,
                                llvm::Twine const &context) {
  // The code goes first: long kernel names in the context get truncated.
  return driver->status.create("ze_result_t 0x" + llvm::Twine::utohexstr(result) +
                               ": " + context);
}

} // namespace M::Target::LevelZero
