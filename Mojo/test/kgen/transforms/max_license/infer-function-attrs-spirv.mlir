// RUN: kgen-opt %s -infer-function-attrs -allow-unregistered-dialect | FileCheck %s

// OpenCL collectives are external calls; only isConvergentOp marks them convergent.
module attributes {M.target_info = #M.target<triple = "spirv64-unknown-unknown", arch = "bmg", features = "", data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", simd_bit_width = 128>} {

  // CHECK: kgen.func @wg_barrier() attributes {convergent}
  kgen.func @wg_barrier() {
    %flags = kgen.param.constant: scalar<ui32> = <3>
    pop.external_call @_Z7barrierj(%flags) : (!kgen.scalar<ui32>) -> ()
    hlcf.return
  }

  // CHECK: kgen.func @ballot() attributes {convergent}
  kgen.func @ballot() {
    %pred = kgen.param.constant: scalar<si32> = <1>
    %m = pop.external_call @_Z16sub_group_balloti(%pred) : (!kgen.scalar<si32>) -> !kgen.scalar<ui32>
    hlcf.return
  }

  // CHECK: kgen.func @caller() attributes {convergent}
  kgen.func @caller() {
    kgen.call @wg_barrier() : () -> ()
    hlcf.return
  }

  // CHECK: kgen.func @lane_id() {
  kgen.func @lane_id() {
    %l = pop.external_call @_Z22get_sub_group_local_idv() : () -> !kgen.scalar<ui32>
    hlcf.return
  }
}
