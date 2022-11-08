func.func @foo(%in: i32) -> () {
    %i1 = sb.constant(5: i32) : !sb.int
    %i2 = sb.constant(7: i32) : !sb.int
    %in_ = builtin.unrealized_conversion_cast %in : i32 to !sb.int
    %i3 = sb.addi %in_ %i1
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