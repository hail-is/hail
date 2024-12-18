package is.hail.lir

object SplitLargeBlocks {
  def splitLargeBlock(m: Method, b: Block): Unit = {
    var L = new Block()
    L.method = m
    var size = 0

    b.replace(L)

    def visit(x: ValueX): Unit = {
      // FIXME this doesn't handle a large number of moderate-sized children
      x.children.foreach(visit)
      size += 1
      if (size > SplitMethod.TargetMethodSize) {
        val l = m.newLocal(genName("l", "split_large_block"), x.ti)
        x.replace(load(l))
        L.append(store(l, x))
        val newL = new Block()
        newL.method = m
        L.append(goto(newL))
        L = newL
        size = 0
      }
    }

    while (b.first != null) {
      val x = b.first
      x.remove()

      x.children.foreach(visit)
      size += 1
      L.append(x)
      if (size > SplitMethod.TargetMethodSize && b.first != null) {
        val newL = new Block()
        newL.method = m
        L.append(goto(newL))
        L = newL
        size = 0
      }
    }
  }

  def apply(m: Method): Unit = {
    val blocks = m.findBlocks()

    for (b <- blocks)
      if (b.approxByteCodeSize() > SplitMethod.TargetMethodSize)
        splitLargeBlock(m, b)
  }
}
