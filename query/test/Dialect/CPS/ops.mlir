// RUN: hail-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: hail-opt %s | hail-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: hail-opt -mlir-print-op-generic %s | hail-opt | FileCheck %s

func.func @callcc(%arg0: i32) -> i32 {
  %result = cps.callcc %ret : i32 {
    %cont = cps.cont(%arg1: i32, %arg2: i64) {
      cps.apply %ret(%arg1) : i32
    }
    %cont2 = cps.cont() {
      %c = arith.constant 0 : i64
      cps.apply %cont(%arg0, %c) : i32, i64
    }
    cps.apply %cont2
  }
  return %result : i32
}
// CHECK-LABEL: func @callcc(
// CHECK-NEXT:    %{{.+}} = cps.callcc %{{.+}} : i32 {
// CHECK-NEXT:      %{{.+}} = cps.cont(%{{.+}}: i32, %{{.+}}: i64) {
// CHECK-NEXT:        cps.apply %{{.+}}(%{{.+}}) : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.+}} = cps.cont() {
// CHECK-NEXT:        %{{.+}} = arith.constant 0 : i64
// CHECK-NEXT:        cps.apply %{{.+}}(%{{.+}}, %{{.+}}) : i32, i64
// CHECK-NEXT:      }
// CHECK-NEXT:      cps.apply %{{.+}}
// CHECK-NEXT:    }
// CHECK-NEXT:    return %{{.+}} : i32
