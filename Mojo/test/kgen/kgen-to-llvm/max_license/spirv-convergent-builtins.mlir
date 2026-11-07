// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

// CHECK: llvm.func @_Z7barrierj(i32) attributes {convergent
// CHECK: llvm.func @_Z22get_sub_group_local_idv() -> i32 attributes {passthrough
module attributes {M.target_info = #M.target<triple = "spirv64-unknown-unknown", arch = "bmg", features = "", data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", simd_bit_width = 128>} {
  kgen.func @wg_barrier() {
    %flags = kgen.param.constant: scalar<ui32> = <3>
    pop.external_call @_Z7barrierj(%flags) : (!kgen.scalar<ui32>) -> ()
    %l = pop.external_call @_Z22get_sub_group_local_idv() : () -> !kgen.scalar<ui32>
    hlcf.return
  }
}
