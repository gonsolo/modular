load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ze_headers",
    hdrs = glob(["include/*.h"]),
    include_prefix = "level_zero",
    strip_include_prefix = "include",
)
