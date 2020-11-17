package is.hail.lir

object InitializeLocals {
  def apply(
    m: Method,
    blocks: Blocks,
    locals: Locals,
    liveness: Liveness
  ): Unit = {
    val entryIdx = blocks.index(m.entry)
    val entryUsedIn = liveness.liveIn(entryIdx)

    var i = entryUsedIn.nextSetBit(0)
    while (i >= 0) {
      val l = locals(i)
      if (!l.isInstanceOf[Parameter]) {
        // println(s"  init $l ${l.ti}")
        m.entry.prepend(
          store(locals(i), defaultValue(l.ti, lineNumber = 0), lineNumber = 0))
      }
      i = entryUsedIn.nextSetBit(i + 1)
    }
  }
}
