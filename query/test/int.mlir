// RUN: hail-opt %s | hail-opt | FileCheck %s
// RUN: hail-opt %s --mlir-print-op-generic | hail-opt | FileCheck %s

// CHECK-LABEL: test_int
func.func @test_int() {
    %b = arith.constant 1 : i1
    %0 = scf.if %b -> !sb.int {
        %1 = sb.constant(5) : !sb.int
        scf.yield %1 : !sb.int
    } else {
        %2 = sb.constant(-2) : !sb.int
        scf.yield %2 : !sb.int
    }
    return
}
