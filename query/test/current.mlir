func.func @foo(%in: !hail.int) -> (!hail.int) {
    %i1 = hail.i32 5 : !hail.int
    %i2 = hail.i32 7 : !hail.int
    %i3 = hail.i32_plus %in %i1
    %i4 = hail.i32_plus %i3 %i2
    func.return %i4 : !hail.int
}