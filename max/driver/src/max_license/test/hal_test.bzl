"""Starlark helper for the Intel Level Zero HAL GPU tests."""

load("//bazel:api.bzl", "mojo_test")

_HAL_TEST_ENV = {
    "MODULAR_DRIVER_PLUGINS": "level_zero@$(rootpath //Target:LevelZeroPlugin.so);cpu@$(rootpath //Target:CPUPlugin.so)",
    "ZE_AFFINITY_MASK": "1",
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
