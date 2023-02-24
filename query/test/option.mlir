// RUN: hail-opt -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: option
func.func @option() -> i32 {
  %0 = option.construct !option.option<i32, i32> {
  ^bb0(%missing: !cps.cont<>, %present: !cps.cont<i32, i32>):
    %y:2 = "init_value1"() : () -> (i32, i32)
    %cond = "cond"() : () -> i1
    cps.if %cond, %present(%y#0, %y#1 : i32, i32), %missing
  }
  %z = "init_value2"() : () -> (i32)
  %1 = option.construct !option.option<i32> {
  ^bb0(%missing: !cps.cont<>, %present: !cps.cont<i32>):
    cps.apply %present(%z) : i32
  }
  %2 = option.map(%0, %1) : (!option.option<i32, i32>, !option.option<i32>) -> (!option.option<i32, i32>) {
  ^bb0(%x1: i32, %x2: i32, %x3: i32):
    %y:2 = "map_body"(%x1, %x2, %x3) : (i32, i32, i32) -> (i32, i32)
    option.yield %y#0, %y#1 : i32, i32
  }
  // "use"(%0) : (!option.option<i32, i32>) -> ()
  %3 = cps.callcc %ret : i32 {
    %missing = cps.cont() {
      %c1 = arith.constant 1 : i32
      cps.apply %ret(%c1) : i32
    }
    %present = cps.cont(%v1: i32, %v2: i32) {
      %r = "consume"(%v1, %v2) : (i32, i32) -> (i32)
      cps.apply %ret(%r) : i32
    }
    option.destruct(%2)[%missing, %present] : !option.option<i32, i32>
  }
  return %3 : i32
}
