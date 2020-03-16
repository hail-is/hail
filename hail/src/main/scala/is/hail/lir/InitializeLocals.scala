package is.hail.lir

import scala.collection.mutable

object InitializeLocals {
  def apply(m: Method): Unit = new InitializeLocals(m).initializeLocals()
}

class InitializeLocals(m: Method) {
  val (blocks, blockIdx) = m.findAndIndexBlocks()
  val (locals, localIdx) = m.findAndIndexLocals(blocks)

  private val cfg = CFG(m, blocks, blockIdx)

  private val nBlocks = blocks.length
  private val nLocals = locals.length

  private val gen: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))
  private val kill: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))

  private val usedIn: Array[java.util.BitSet] = Array.fill(nBlocks)(new java.util.BitSet(nLocals))

  private def computeGenKill(): Unit = {
    var i = 0
    while (i < nBlocks) {
      val b = blocks(i)

      // cache
      val geni = gen(i)
      val killi = kill(i)

      def visit(x: X): Unit = {
        x match {
          case x: StoreX =>
            val l = localIdx(x.l)
            geni.clear(l)
            killi.set(l)
          case x: LoadX =>
            val l = localIdx(x.l)
            geni.set(l)
          case x: IincX =>
            val l = localIdx(x.l)
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

  private def computeUsedIn(): Unit = {
    val q = mutable.Set[Int]()

    (0 until nBlocks).foreach(q += _)

    while (q.nonEmpty) {
      val i = q.head
      q -= i

      val newUsedIn = new java.util.BitSet(nLocals)
      for (j <- cfg.succ(i))
        newUsedIn.or(usedIn(j))
      newUsedIn.andNot(kill(i))
      newUsedIn.or(gen(i))

      if (newUsedIn != usedIn(i)) {
        usedIn(i) = newUsedIn
        for (j <- cfg.pred(i))
          q += j
      }
    }

    /*
    // dump usedIn
    for (b <- blocks) {
      val i = blockIdx(b)
      val usedIni = usedIn(i)

      print(s"$i/$b:")

      var j = usedIni.nextSetBit(0)
      while (j >= 0) {
        val l = locals(j)
        print(s" $i/$l")
        j = usedIni.nextSetBit(j + 1)
      }
      println()
    }
    */
  }

  def initializeLocals(): Unit = {
    computeGenKill()
    computeUsedIn()

    val entryIdx = blockIdx(m.entry)
    val entryUsedIn = usedIn(entryIdx)

    var i = entryUsedIn.nextSetBit(0)
    while (i >= 0) {
      val l = locals(i)
      if (!l.isInstanceOf[Parameter]) {
        // println(s"  init $l ${l.ti}")
        m.entry.prepend(
          store(locals(i), defaultValue(l.ti)))
      }

      i = entryUsedIn.nextSetBit(i + 1)
    }
  }
}
