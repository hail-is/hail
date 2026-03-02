package is.hail.lir

import scala.collection.mutable

object CFG {
  def apply(m: Method, blocks: Blocks): CFG = {
    val nBlocks = blocks.nBlocks

    val pred = Array.fill(nBlocks)(mutable.Set[Int]())
    val succ = Array.fill(nBlocks)(mutable.Set[Int]())

    for (i <- blocks.indices) {

      def edgeTo(L: Block): Unit = {
        val j = blocks.index(L)

        succ(i) += j
        pred(j) += i
      }

      val x = blocks(i).last.asInstanceOf[ControlX]
      for (i <- x.targetIndices)
        edgeTo(x.target(i))
    }

    new CFG(blocks.index(m.entry), pred, succ)
  }
}

class CFG(
  val entry: Int,
  val pred: Array[mutable.Set[Int]],
  val succ: Array[mutable.Set[Int]],
) {
  def nBlocks: Int = succ.length

  def dump(): Unit = {
    println(s"CFG $nBlocks:")
    var i = 0
    while (i < nBlocks) {
      println(s"  $i: ${succ(i).mkString(",")}")
      i += 1
    }
  }
}
