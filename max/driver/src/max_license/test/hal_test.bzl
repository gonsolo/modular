"""Starlark helper for the Intel Level Zero HAL GPU tests."""

load("//bazel:api.bzl", "mojo_test")

_HAL_TEST_ENV = {
    "MODULAR_DRIVER_PLUGINS": "level_zero@$(rootpath //Target:LevelZeroPlugin.so);cpu@$(rootpath //Target:CPUPlugin.so)",
    "ZE_AFFINITY_MASK": "1",
}

# Bucket D (multi-device) needs BOTH B60s visible, so the affinity mask is dropped
# (unset -> the driver reports all devices). On this box that is devices 0 and 1.
_HAL_TEST_ENV_MULTI = {
    "MODULAR_DRIVER_PLUGINS": "level_zero@$(rootpath //Target:LevelZeroPlugin.so);cpu@$(rootpath //Target:CPUPlugin.so)",
}

_HAL_TEST_DATA = [
    "//Target:LevelZeroPlugin.so",
    "//Target:CPUPlugin.so",
]

_HAL_TEST_DEPS = [
    "//max/driver/src/max_license:max",
    "@mojo//:std",
    "@mojo//:test_utils",
]

def hal_gpu_test(name):
    """Declares a `name.mojo.test` HAL device-execution test for the B60."""
    mojo_test(
        name = name + ".mojo.test",
        srcs = [name + ".mojo"],
        copts = [
            # Compile the kernels for the Intel Xe2 (Battlemage) target.
            "--target-accelerator=intel:xe2",
        ],
        tags = ["gpu", "manual"],
        env = _HAL_TEST_ENV,
        data = _HAL_TEST_DATA,
        deps = _HAL_TEST_DEPS,
    )

def hal_gpu_multi_test(name):
    """Declares a multi-device (Bucket D) HAL test with BOTH B60s visible.

    Same as `hal_gpu_test` but with `ZE_AFFINITY_MASK` unset so the driver reports
    all devices (both B60s), which `DeviceContext(device_id=0)`/`(1)` then select.
    """
    mojo_test(
        name = name + ".mojo.test",
        srcs = [name + ".mojo"],
        copts = [
            # Compile the kernels for the Intel Xe2 (Battlemage) target.
            "--target-accelerator=intel:xe2",
        ],
        tags = ["gpu", "manual"],
        env = _HAL_TEST_ENV_MULTI,
        data = _HAL_TEST_DATA,
        deps = _HAL_TEST_DEPS,
    )

# Bucket D tier 3 (collectives): like hal_gpu_multi_test but also links the real
# `comm` library + `layout` (the host-staged allreduce / tensor primitives).
_HAL_MULTI_COMM_DEPS = _HAL_TEST_DEPS + [
    "//max/kernels/src/comm:comm",
    "//max:layout",
]

def hal_gpu_multi_comm_test(name):
    """A multi-device (Bucket D) collective test: both B60s + the comm library."""
    mojo_test(
        name = name + ".mojo.test",
        srcs = [name + ".mojo"],
        copts = [
            "--target-accelerator=intel:xe2",
        ],
        tags = ["gpu", "manual"],
        env = _HAL_TEST_ENV_MULTI,
        data = _HAL_TEST_DATA,
        deps = _HAL_MULTI_COMM_DEPS,
    )
