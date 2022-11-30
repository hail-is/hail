// RUN: hail-opt %s | FileCheck %s

// CHECK-LABEL: main
func.func @main() {
  %foo = cps.callcc %ret : i32 {
    %cont = cps.cont(%arg1: i32) {
      cps.apply %ret(%arg1) : i32
    }
    %cont2 = cps.cont() {
      %x = arith.constant 0 : i32
      cps.apply %cont(%x) : i32
    }
    cps.apply %cont2
  }
  func.return
}
