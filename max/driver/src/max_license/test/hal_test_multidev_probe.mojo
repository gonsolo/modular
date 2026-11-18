"""Bucket D, tier 1: multi-device enumeration + per-device context + P2P query.

This is a *probe*, not a collective. It confirms the foundation that the rest of
Bucket D (host-staged collectives) builds on, on this box's dual-B60 (one physical
card, two B60s reported as devices 0 and 1):

  1. `DeviceContext.number_of_devices()` reports **2** devices (i.e. the affinity
     mask is dropped so BOTH B60s are visible to the driver).
  2. We can construct **two** distinct `DeviceContext`s: `DeviceContext(0)` and
     `DeviceContext(1)` -- per-device context selection through the HAL fork.
  3. We can query P2P between them (`ctx0.can_access(ctx1)` / the reverse). The
     expected answer on THIS hardware is **False**: the two B60s sit on separate
     PCIe switch ports (see the PCIe topology note below), so there is no direct
     P2P interconnect. That is the *correct* hardware answer, not a bug -- and it
     is what tells us to use the host-staged (non-P2P) collective path for tier 3.

This test only deps on the HAL `max` + `std` (minimal-HAL style). It is run with
`ZE_AFFINITY_MASK` unset (see the D variant in hal_test.bzl) so both B60s show up.

PCIe topology of this box (the reason P2P is expected False):
  B60 #0 (card3): pci0000:80 / 80:01.1 / 81:00.0 / 82:01.0 / 83:00.0
  B60 #1 (card4): pci0000:80 / 80:01.3 / 85:00.0 / 86:01.0 / 87:00.0
  -> they branch at 80:01.1 vs 80:01.3 (two separate PCIe switch ports under the
     same root), each behind its own bridge chain. No direct link between the two
     B60s, so one device cannot DMA straight into the other's memory.

Run via Bazel:  //max/driver/src/max_license/test:hal_test_multidev_probe.mojo.test
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext


def main() raises:
    # --- 1. Both B60s must be visible (affinity mask dropped). -----------------
    var n = DeviceContext.number_of_devices()
    print("number_of_devices() =", n)
    assert_true(n >= 2, "expected >= 2 devices (both B60s) with ZE_AFFINITY_MASK unset")

    # --- 2. Per-device context selection: two distinct contexts. --------------
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    print("constructed DeviceContext(device_id=0) and DeviceContext(device_id=1)")

    # Basic per-device liveness: each context can report its own memory info
    # without error (a cheap, real driver call that proves both contexts are
    # backed by a live device, not a stub).
    var (free0, total0) = ctx0.get_memory_info()
    var (free1, total1) = ctx1.get_memory_info()
    print(t"ctx0 memory: free={free0} total={total0}")
    print(t"ctx1 memory: free={free1} total={total1}")
    assert_true(total0 > 0, "ctx0 should report non-zero total memory")
    assert_true(total1 > 0, "ctx1 should report non-zero total memory")

    # --- 3. P2P capability query (expected False on this box). -----------------
    var can01 = ctx0.can_access(ctx1)
    var can10 = ctx1.can_access(ctx0)
    print("ctx0.can_access(ctx1) =", can01)
    print("ctx1.can_access(ctx0) =", can10)

    # We do NOT assert a specific P2P value: it is a hardware property. On this
    # box it is expected False (separate PCIe switch ports), which routes Bucket D
    # to the host-staged collective path (tier 3). If it ever reports True on a
    # box with a real P2P interconnect, the P2P collective path becomes available.
    print("P2P probe complete.")
