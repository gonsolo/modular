# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""`test_dim` run through the HAL `DeviceContext`.


This is a **pure host-side** test — no kernel is launched and no GPU is
required. It exercises the `Dim` type (the launch-config dimension triple used
by `grid_dim=` / `block_dim=`) and its string formatting. It validates the
host-side launch-config plumbing that underpins every other HAL test, and it
cannot hit the in-kernel `grid_dim` read gap (upstream #7505) because it never
reads `grid_dim` inside a kernel.
"""

from max.gpu.host import Dim
from std.testing import assert_equal


def test_dim() raises:
    print("== test_dim")

    assert_equal(String(Dim(4, 1, 2)), "(x=4, y=1, z=2)")
    assert_equal(String(Dim(4, 2)), "(x=4, y=2)")
    assert_equal(String(Dim(4)), "(x=4, )")
    assert_equal(String(Dim((4, 5))), "(x=4, y=5)")
    assert_equal(String(Dim((4, 2, 3))), "(x=4, y=2, z=3)")


def main() raises:
    test_dim()
