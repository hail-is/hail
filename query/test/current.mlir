func.func @foo(%in: !sb.int) -> (!sb.bool) {
    %i1 = sb.constant 5 : !sb.int
    %i2 = sb.constant 7 : !sb.int
    %i3 = sb.addi %in %i1
    %i4 = sb.addi %i3 %i2
    %i5 = sb.constant 6 : !sb.int
    %i6 = sb.compare eq, %i4, %i5 : !sb.bool
    func.return %i6 : !sb.bool
}