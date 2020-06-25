package is.hail.lir

import scala.collection.mutable

object Liveness {
  def apply(
    blocks: Array[Block],
    blockIdx: Map[Block, Int],
    locals: Array[Local],
    localIdx: Map[Local, Int],
    cfg: CFG
  ): Liveness = {
    val nBlocks = blocks.length

    val nLocals = locals.length

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
              if (!x.l.isInstanceOf[Parameter]) {
                val l = localIdx(x.l)
                geni.clear(l)
                killi.set(l)
              }
            case x: LoadX =>
              if (!x.l.isInstanceOf[Parameter]) {
                val l = localIdx(x.l)
                geni.set(l)
              }
            case x: IincX =>
              if (!x.l.isInstanceOf[Parameter]) {
                val l = localIdx(x.l)
                geni.set(l)
                killi.set(l)
              }
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
      val q = mutable.Set[Int]()

      (0 until nBlocks).foreach(q += _)

      while (q.nonEmpty) {
        val i = q.head
        q -= i

        val enwLiveIn = new java.util.BitSet(nLocals)
        for (j <- cfg.succ(i))
          enwLiveIn.or(liveIn(j))
        enwLiveIn.andNot(kill(i))
        enwLiveIn.or(gen(i))

        if (enwLiveIn != liveIn(i)) {
          liveIn(i) = enwLiveIn
          for (j <- cfg.pred(i))
            q += j
        }
      }
    }

    computeGenKill()
    computeLiveIn()

    new Liveness(liveIn)
  }
}

class Liveness(
  val liveIn: Array[java.util.BitSet])
