%b = arith.constant 1 : i1
%0 = scf.if %b -> !sb.int {
    %1 = sb.constant 5 : !sb.int
    scf.yield %1 : !sb.int
} else {
    %2 = sb.constant -2 : !sb.int
    scf.yield %2 : !sb.int
}
