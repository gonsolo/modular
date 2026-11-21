"""Starlark helper for the Intel Level Zero HAL GPU tests.

The suite pins to the Intel Xe2 (Battlemage / B60) target via `--target-accelerator`.

Device selection
----------------
`ZE_AFFINITY_MASK` is intentionally NOT hard-coded. When unset the Level Zero driver
reports every visible GPU; when a single-B60 machine is the target that is the natural
default, and on a multi-GPU box the test harness (or a CI variable) can set
`ZE_AFFINITY_MASK` to select a specific device via `--test_env=ZE_AFFINITY_MASK=<mask>`.
Hard-coding `ZE_AFFINITY_MASK=1` in the rule would (a) hide a second device on a
multi-GPU box and (b) make the suite non-portable to other Intel targets, so the mask
is left to the environment.

Multi-device (Bucket D) tests live in a separate PR (held on modular/modular#7623);
the `hal_gpu_multi_*` macros are therefore not declared here.
"""

load("//bazel:api.bzl", "mojo_test")

# The only env the suite needs: which driver plugins to load. `ZE_AFFINITY_MASK` is
# deliberately omitted so it can be set from the environment / CI per machine.
_HAL_TEST_ENV = {
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
    """Declares a `name.mojo.test` HAL device-execution test for the Intel Xe2 target.

    `ZE_AFFINITY_MASK` is read from the environment (unset by default) so the test runs
    on whatever device(s) the driver reports; override with
    `--test_env=ZE_AFFINITY_MASK=<mask>` to pin a specific GPU.
    """
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
