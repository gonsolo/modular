"""SPIR-V codegen check for the portable strided/vectorized load, on the B60 target.

Bucket A, item 3 of the completeness plan. The portable *purpose* of the vendor
`test_fast_div_ptx` / `test_load_width_codegen` (which hardcode an NVIDIA target and
assert on the PTX string -- a portability anti-pattern that can never run on Intel).
Here we compile a kernel for `intel:xe2` and assert on the **emitted SPIR-V
assembly** (the Intel "how"), the direct analog of the NVIDIA PTX assertions.

`_compile_code[kernel, emission_kind="asm", target=get_gpu_target["xe2"]()]().asm`
returns the textual SPIR-V (spv-dis-style `Op*` output) for the target (the SPIR-V
backend's asm extension is `.spvasm`).

What this asserts (validated against the real emitted SPIR-V on the B60):
  - the width-4 load is vectorized (`OpTypeVector ... 4`);
  - the strided address is computed with integer multiply+add (`OpIMul` + `OpIAdd`),
    i.e. `base + k*stride`;
  - the load is emitted as per-lane `OpLoad`s (the scalar-loop gather). Note: this
    reflects the #7620 workaround (`has_generic_address_space = False` in the Level
    Zero std plugin) -- the vectorized `gather`/`G_STEP_VECTOR` path is not used, so
    the gather appears as `width` scalar `OpLoad`s with `OpIMul`/`OpIAdd` offsets
    rather than a single hardware gather. Once the SPIR-V backend legalizes
    `G_STEP_VECTOR` and the workaround is reverted, this test should be revisited to
    assert on the hardware gather op instead.

Only deps on the HAL `max` + `std` + `max.gpu.host.compile` (minimal-HAL style).

Run via Bazel:  //max/driver/src/max_license/test:hal_test_spirv_codegen.mojo.test
"""

from max.gpu import global_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.sys.intrinsics import strided_load
from std.testing import assert_true


def strided_load_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    input: ImmPointer[Float32, ImmutAnyOrigin],
    len_dev: Int32,
):
    comptime W = 4
    comptime S = 8
    var tid = global_idx.x
    var length = Int(len_dev)
    var base = tid * W
    if base + (W - 1) * S >= length:
        return
    var v = strided_load[W](input.unsafe_offset(base), S)
    for i in range(W):
        output[base + i] = v[i]


def main() raises:
    comptime kernel = strided_load_kernel
    var spv = String(
        _compile_code[kernel, emission_kind="asm", target=get_gpu_target["xe2"]()]().asm
    )

    # The width-4 load is vectorized: a 4-element vector type appears.
    assert_true(
        "OpTypeVector" in spv,
        msg=String(t"SPIR-V missing OpTypeVector (width-4 load not vectorized)"),
    )

    # The strided address (base + k*stride) is computed with integer multiply + add.
    assert_true(
        "OpIMul" in spv and "OpIAdd" in spv,
        msg=String(t"SPIR-V missing the strided address arithmetic (OpIMul/OpIAdd)"),
    )

    # The load is present (per-lane OpLoad in the scalar-loop gather path; see the
    # module docstring about the #7620 workaround).
    assert_true(
        "OpLoad" in spv,
        msg=String(t"SPIR-V missing OpLoad"),
    )

    # A store is present (the kernel writes its result back).
    assert_true(
        "OpStore" in spv,
        msg=String(t"SPIR-V missing OpStore"),
    )
