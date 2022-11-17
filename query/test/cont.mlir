%foo = cps.callcc %ret : i32 {
  %cont = cps.cont(%arg1: i32, %cont: !cps.cont<i32>) {
    cps.apply %cont(%arg1) : i32
  }
  %cont2 = cps.cont() {
    %x = arith.constant 0 : i32
    cps.apply %cont(%x, %ret) : i32, !cps.cont<i32>
  }
  cps.apply %cont2()
}