package is.hail.expr.ir.analyses

import is.hail.expr.ir._

object CompilationUnitSize {
  def apply(ir: IR): Memo[Int] = {
    val m = Memo.empty[Int]
    def recur(x: IR): Int = {

      val childrenSum = x.children.iterator.map { case child: IR => recur(child) }.sum
      val currentSize = x match {
        case _: CollectDistributedArray => 1
        case _ => childrenSum
      }
      m.bind(x, currentSize)
      currentSize
    }
    recur(ir)
    m
  }
}
