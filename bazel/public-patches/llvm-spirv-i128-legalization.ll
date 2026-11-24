; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that scalar i128 operations are legalized (narrowed to i64 pairs)
; for SPIR-V targets instead of failing selection/legalization.

; CHECK-DAG: OpTypeInt 64 0

define void @add_i128(ptr %pa, ptr %pb, ptr %pr) {
; CHECK-LABEL: OpFunction
  %a = load i128, ptr %pa
  %b = load i128, ptr %pb
  %r = add i128 %a, %b
  store i128 %r, ptr %pr
  ret void
}

define void @sub_mul_i128(ptr %pa, ptr %pb, ptr %pr) {
; CHECK-LABEL: OpFunction
  %a = load i128, ptr %pa
  %b = load i128, ptr %pb
  %s = sub i128 %a, %b
  %m = mul i128 %s, %b
  store i128 %m, ptr %pr
  ret void
}

define void @icmp_i128(ptr %pa, ptr %pb, ptr %pr) {
; CHECK-LABEL: OpFunction
  %a = load i128, ptr %pa
  %b = load i128, ptr %pb
  %c = icmp ult i128 %a, %b
  %z = zext i1 %c to i32
  store i32 %z, ptr %pr
  ret void
}

define void @zext_trunc_i128(ptr %pa, ptr %pr) {
; CHECK-LABEL: OpFunction
  %a = load i64, ptr %pa
  %z = zext i64 %a to i128
  %t = trunc i128 %z to i64
  store i64 %t, ptr %pr
  ret void
}

define void @select_i128(ptr %pc, ptr %pa, ptr %pb, ptr %pr) {
; CHECK-LABEL: OpFunction
  %cv = load i32, ptr %pc
  %c = trunc i32 %cv to i1
  %a = load i128, ptr %pa
  %b = load i128, ptr %pb
  %r = select i1 %c, i128 %a, i128 %b
  store i128 %r, ptr %pr
  ret void
}
