package is.hail.lir

import scala.collection.mutable

object CFG {
  def apply(m: Method, blocks: Blocks): CFG = {
    val nBlocks = blocks.nBlocks

    val pred = Array.fill(nBlocks)(mutable.Set[Int]())
    val succ = Array.fill(nBlocks)(mutable.Set[Int]())

    val N = blocks.length
    var i = 0
    while (i < N) {
      val x = blocks(i).last.asInstanceOf[ControlX]

      val T = x.targetArity()
      var t = 0
      while (t < T) {
        val j = blocks.index(x.target(t))
        succ(i) += j
        pred(j) += i
        t += 1
      }

      i += 1
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
