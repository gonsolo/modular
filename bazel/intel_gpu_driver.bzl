"""Intel's GPU driver stack built from pinned sources, for `--target-accelerator intel:xe2`.

IGC, gmmlib and the compute runtime are fetched at pinned versions, with the
fixes the Intel target still needs applied, and built with CMake into the
repository. IGC is the master commit the fixes are tested on, with the
dependency revisions that build used: release v2.42.6 crashes compiling large
kernels (a null source in a struct copy). Move to the next IGC release once it
is out. The compute runtime is release 26.40.40196.8. Opt-in: nothing is fetched
or built unless something depends on @intel_gpu_driver. The first build takes
about an hour (IGC builds its own LLVM). Drop a patch once its fix is released.
"""

_SOURCES = [
    # (directory, url, sha256, strip_prefix)
    ("igc", "https://github.com/intel/intel-graphics-compiler/archive/8850fdf32648425be86a0ab6d9264a8060f228ef.tar.gz", "c9845780acf0e7b18196f9e9f9cd5ce8d14fcb875d32450c0134974015be4201", "intel-graphics-compiler-8850fdf32648425be86a0ab6d9264a8060f228ef"),
    ("vc-intrinsics", "https://github.com/intel/vc-intrinsics/archive/refs/tags/v0.26.0.tar.gz", "a633eedc02db01ad1923b1ab62bc403a42518a666ab2ed8be7251b6958e58d55", "vc-intrinsics-0.26.0"),
    ("SPIRV-Tools", "https://github.com/KhronosGroup/SPIRV-Tools/archive/28f99f74a2a922ff85fc8fd966d6b3d6a45be795.tar.gz", "5aeda0987764f383675b6c99adc7ca9cf63795e9358a575d121f7e27e3803efe", "SPIRV-Tools-28f99f74a2a922ff85fc8fd966d6b3d6a45be795"),
    ("SPIRV-Headers", "https://github.com/KhronosGroup/SPIRV-Headers/archive/fbf86836a9b1966586faf7be7bb8a6440e156d62.tar.gz", "25198fd01717ec1d9b2b260f144fecba1265c51dcc8531ef98008e7b4dbb7e44", "SPIRV-Headers-fbf86836a9b1966586faf7be7bb8a6440e156d62"),
    ("llvm-project", "https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/llvm-project-22.1.8.src.tar.xz", "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888", "llvm-project-22.1.8.src"),
    ("llvm-project/llvm/projects/opencl-clang", "https://github.com/intel/opencl-clang/archive/b953eceebb0abc6ea954a14545420e3f97540a77.tar.gz", "611725f564fc0686518f44a486962af28df0facb151c5192e8ee20d80f2cf429", "opencl-clang-b953eceebb0abc6ea954a14545420e3f97540a77"),
    ("llvm-project/llvm/projects/llvm-spirv", "https://github.com/KhronosGroup/SPIRV-LLVM-Translator/archive/c72cf5cbab17419f410df70f4f0822bf3d26d0fe.tar.gz", "455562c9f605a140d80db6cfabb363014b8de05c5b4663eb8a5513eeadeb64c7", "SPIRV-LLVM-Translator-c72cf5cbab17419f410df70f4f0822bf3d26d0fe"),
    ("gmmlib", "https://github.com/intel/gmmlib/archive/refs/tags/intel-gmmlib-22.10.2.tar.gz", "94d0038ef92bb65b86d08f004197c56f8d81f82f9ac7a25c202a88afd653a30b", "gmmlib-intel-gmmlib-22.10.2"),
    ("compute-runtime", "https://github.com/intel/compute-runtime/archive/refs/tags/26.40.40196.8.tar.gz", "4e543ecef2d2842b42811795120bc3e2f4b8d17f088bb87456104653ab5e274f", "compute-runtime-26.40.40196.8"),
]

_PATCHES = [
    # (directory, patch in //bazel/public-patches)
    ("igc", "igc-augmentation-binary-search.patch"),  # intel/intel-graphics-compiler#460
    ("igc", "igc-struct-return-value.patch"),  # intel/intel-graphics-compiler#463
    ("igc", "igc-struct-phi-variable.patch"),  # intel/intel-graphics-compiler#465
    ("igc", "igc-masked-gather-scatter-input.patch"),  # intel/intel-graphics-compiler#477
    ("igc", "igc-no-remat-stack-call-args.patch"),  # intel/intel-graphics-compiler#467
    ("igc", "igc-split-on-spill-preheader.patch"),  # intel/intel-graphics-compiler#468
    ("igc", "igc-defined-members-nonconstant-base.patch"),  # intel/intel-graphics-compiler#478
    ("igc", "igc-no-werror.patch"),  # newer host compilers warn where IGC's -Werror expects silence
    ("compute-runtime", "compute-runtime-shared-private-memory.patch"),  # intel/compute-runtime#1037
]

_BUILD = """
package(default_visibility = ["//visibility:public"])

# The driver libraries, and env.sh, which points the Level Zero loader at them.
filegroup(
    name = "driver",
    srcs = glob(["install/lib/**"]) + ["env.sh"],
)
"""

def _intel_gpu_driver_impl(rctx):
    for directory, url, sha256, strip_prefix in _SOURCES:
        rctx.download_and_extract(url, output = "src/" + directory, sha256 = sha256, strip_prefix = strip_prefix)
    patches = [("src/" + d, str(rctx.path(Label("//bazel/public-patches:" + p)))) for d, p in _PATCHES]

    # opencl-clang patches clang (OpenCL 3.1 among others) with `git am`, which needs a git checkout; apply them here.
    # Its LLVM patch is left out: IGC applies the same fix to its own copy of LLVM.
    cclang = rctx.path("src/llvm-project/llvm/projects/opencl-clang/patches/clang")
    patches += [("src/llvm-project", str(p)) for p in sorted(cclang.readdir(), key = str)]
    for directory, patch in patches:
        result = rctx.execute(["patch", "-d", directory, "-p1", "-i", patch])
        if result.return_code:
            fail("intel_gpu_driver: {} does not apply:\n{}{}".format(patch, result.stdout, result.stderr))
    script = rctx.path(Label("//bazel/public-patches:intel_gpu_driver_build.sh"))
    install = str(rctx.path("install"))
    result = rctx.execute(["bash", str(script), str(rctx.path("src")), install], timeout = 6 * 3600, quiet = False)
    if result.return_code:
        log = rctx.read(install + "/build.log").splitlines()
        fail("intel_gpu_driver: build failed, the end of its log:\n" + "\n".join(log[-40:]))
    rctx.delete("src")
    rctx.file("env.sh", "\n".join([
        "# Source this to run Mojo programs on the driver built here.",
        "export ZE_ENABLE_ALT_DRIVERS={}/lib/libze_intel_gpu.so.1".format(install),
        "export LD_LIBRARY_PATH={}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}".format(install),
        "",
    ]))
    rctx.file("BUILD.bazel", _BUILD)

intel_gpu_driver = repository_rule(
    implementation = _intel_gpu_driver_impl,
    doc = "IGC, gmmlib and the compute runtime from pinned, patched sources.",
)
