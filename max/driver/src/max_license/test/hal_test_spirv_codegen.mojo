"""SPIR-V codegen check on the Intel Xe2 target.

Portable purpose of vendor PTX-string codegen assertions: compile a small kernel
for `intel:xe2` and assert on emitted SPIR-V assembly (`Op*` mnemonics). This
checks the Intel "how" (SPIR-V) rather than hardcoding NVIDIA PTX.

The strided/gather load codegen case is held on modular/modular#7620
(`G_STEP_VECTOR` not legalizable by the SPIR-V/IGC GISel backend), so this test
uses a simple scalar vector-add kernel instead.
"""

from max.gpu import global_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import assert_true


def vec_add_kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    a: ImmPointer[Float32, ImmutAnyOrigin],
    b: ImmPointer[Float32, ImmutAnyOrigin],
    n: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(n):
        output.unsafe_offset(i)[] = a.unsafe_offset(i)[] + b.unsafe_offset(i)[]


def main() raises:
    var spv = String(
        _compile_code[
            vec_add_kernel, emission_kind="asm", target=get_gpu_target["xe2"]()
        ]().asm
    )

    assert_true("OpTypeFloat" in spv, msg=String(t"SPIR-V missing OpTypeFloat"))
    assert_true("OpLoad" in spv, msg=String(t"SPIR-V missing OpLoad"))
    assert_true("OpFAdd" in spv, msg=String(t"SPIR-V missing OpFAdd"))
    assert_true("OpStore" in spv, msg=String(t"SPIR-V missing OpStore"))
