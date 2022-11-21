func.func @option() -> i32 {
  %0 = option.construct !option.option<i32, i32> {
  ^bb0(%missing: !cps.cont<>, %present: !cps.cont<i32, i32>):
    cps.apply %missing
  }
  %1 = cps.callcc %ret : i32 {
    %missing = cps.cont() {
      %c = arith.constant 1 : i32
      cps.apply %ret(%c) : i32
    }
    %present = cps.cont(%v1: i32, %v2: i32) {
      cps.apply %ret(%v1) : i32
    }
    option.destruct %0(%missing, %present) : !option.option<i32, i32>
  }
  return %1 : i32
}