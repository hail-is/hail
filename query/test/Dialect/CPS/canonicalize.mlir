// RUN: hail-opt %s -pass-pipeline='func.func(canonicalize)' -split-input-file -allow-unregistered-dialect | FileCheck %s


// -----

// CHECK-LABEL:  func @inline_def(
// CHECK:          %[[val:[[:alnum:]]+]] = "init_val"()
// CHECK-NEXT:     %[[x:[[:alnum:]]+]] = "cont_body"(%[[val]])
func.func @inline_def() -> i32 {
  %result = cps.callcc %ret : i32 {
    %cont = cps.cont(%arg: i32) {
      %x = "cont_body"(%arg) : (i32) -> i32
      cps.apply %ret(%x) : i32
    }
    %val = "init_val"() : () -> i32
    cps.apply %cont(%val) : i32
  }
  return %result : i32
}

// -----

// CHECK-LABEL:  func @trivial_callcc(
// CHECK-NEXT:     %[[val:[[:alnum:]]+]] = "compute_val"()
// CHECK-NEXT:     return %[[val]]
func.func @trivial_callcc() -> i32 {
  %result = cps.callcc %ret : i32 {
    %val = "compute_val"() : () -> i32
    cps.apply %ret(%val) : i32
  }
  return %result : i32
}
