%b = arith.constant 1 : i1
%0 = scf.if %b -> !hail.int {
    %1 = hail.i32 5 : !hail.int
    scf.yield %1 : !hail.int
} else {
    %2 = hail.i32 -2 : !hail.int
    scf.yield %2 : !hail.int
}
