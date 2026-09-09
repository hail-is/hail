package is.hail.lir

object Liveness {
  def apply(
    blocks: Blocks,
    locals: Locals,
    cfg: CFG,
  ): Liveness = {
    val nBlocks = blocks.nBlocks

    val nLocals = locals.nLocals

    val gen: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))
    val kill: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))

    val liveIn: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))

    def computeGenKill(): Unit = {
      var i = 0
      while (i < nBlocks) {
        val b = blocks(i)

        // cache
        val geni = gen(i)
        val killi = kill(i)

        def visit(x: X): Unit = {
          x match {
            case x: StoreX =>
              val l = locals.index(x.l)
              geni.clear(l)
              killi.set(l)
            case x: LoadX =>
              val l = locals.index(x.l)
              geni.set(l)
            case x: IincX =>
              val l = locals.index(x.l)
              geni.set(l)
              killi.set(l)
            case _ =>
          }
          x.children.foreach(visit)
        }

        var x = b.last
        while (x != null) {
          visit(x)
          x = x.prev
        }

        i += 1
      }
    }

    def computeLiveIn(): Unit = {
      val q = new java.util.BitSet(nBlocks)
      q.set(0, nBlocks)

      while (!q.isEmpty) {
        // liveness is a backward analysis and block indices are in reverse post-order
        // (see `Method.findBlocks`). Draining the largest-first visits successors before
        // predecessors and converges in fewer iterations.
        val i = q.previousSetBit(nBlocks - 1)
        q.clear(i)

        val newLiveIn = new java.util.BitSet(nLocals)
        for (j <- cfg.succ(i))
          newLiveIn.or(liveIn(j))
        newLiveIn.andNot(kill(i))
        newLiveIn.or(gen(i))

        if (newLiveIn != liveIn(i)) {
          liveIn(i) = newLiveIn
          for (j <- cfg.pred(i))
            q.set(j)
        }
      }
    }

    computeGenKill()
    computeLiveIn()

    new Liveness(liveIn)
  }
}

class Liveness(
  val liveIn: Array[java.util.BitSet]
)
