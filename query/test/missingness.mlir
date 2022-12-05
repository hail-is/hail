// RUN: hail-opt %s
func.func @loop_inner_control_flow(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  %cst_1 = arith.constant 1 : i32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %cst_1) -> (i32) {
    %cond = missing.is_missing %si : i32
    %inner_res = scf.if %cond -> (i32) {
      %1 = missing.missing : i32
      scf.yield %1 : i32
    } else {
      %si_inc = arith.addi %si, %cst_1 : i32
      scf.yield %si_inc : i32
    }
    scf.yield %inner_res : i32
  }
  return %result : i32
}
