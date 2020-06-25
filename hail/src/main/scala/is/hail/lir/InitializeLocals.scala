package is.hail.lir

object InitializeLocals {
  def apply(
    m: Method,
    blocks: Array[Block],
    blockIdx: Map[Block, Int],
    locals: Array[Local],
    liveness: Liveness
  ): Unit = {
    val entryIdx = blockIdx(m.entry)
    val entryUsedIn = liveness.liveIn(entryIdx)

    var i = entryUsedIn.nextSetBit(0)
    while (i >= 0) {
      val l = locals(i)
      // println(s"  init $l ${l.ti}")
      m.entry.prepend(
        store(locals(i), defaultValue(l.ti)))

      i = entryUsedIn.nextSetBit(i + 1)
    }
  }
}
