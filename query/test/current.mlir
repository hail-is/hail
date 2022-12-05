// RUN: hail-opt %s
func.func @main() -> () {
    %in = sb.constant(1: i32) : !sb.int
    %i1 = sb.constant(5: i32) : !sb.int
    %i2 = sb.constant(7: i32) : !sb.int
    %i3 = sb.addi %in %i1
    %i4 = sb.addi %i3 %i2
    %i5 = sb.constant(6: i32) : !sb.int
    %i6 = sb.compare eq, %i4, %i5 : !sb.bool
    sb.print %i6 : !sb.bool
    func.return
}

func.func @bar() -> () {
    %i1 = sb.constant(true) : !sb.bool
    sb.print %i1 : !sb.bool
    func.return
}
