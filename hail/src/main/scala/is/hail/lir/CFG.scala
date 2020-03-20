package is.hail.lir

import scala.collection.mutable

object CFG {
  def apply(m: Method, blocks: IndexedSeq[Block], blockIdx: Map[Block, Int]): CFG = {
    val nBlocks = blocks.length

    val pred = Array.fill(nBlocks)(mutable.Set[Int]())
    val succ = Array.fill(nBlocks)(mutable.Set[Int]())

    for (b <- blocks) {
      val i = blockIdx(b)

      def edgeTo(L: Block): Unit = {
        val j = blockIdx(L)
        succ(i) += j
        pred(j) += i
      }

      b.last match {
        case x: GotoX => edgeTo(x.L)
        case x: IfX =>
          edgeTo(x.Ltrue)
          edgeTo(x.Lfalse)
        case x: SwitchX =>
          edgeTo(x.Ldefault)
          x.Lcases.foreach(edgeTo)
        case x: ReturnX =>
        case x: ThrowX =>
      }
    }

    new CFG(pred, succ)
  }
}
class CFG(
  val pred: Array[mutable.Set[Int]],
  val succ: Array[mutable.Set[Int]])
