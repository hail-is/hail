func.func @foo(%in: !sb.int) -> (!sb.int) {
    %i1 = sb.constant 5 : !sb.int
    %i2 = sb.constant 7 : !sb.int
    %i3 = sb.addi %in %i1
    %i4 = sb.addi %i3 %i2
    func.return %i4 : !sb.int
}