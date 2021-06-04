package is.hail.expr.ir.analyses

import is.hail.HailContext
import is.hail.expr.ir._

object ComputeMethodSplits {
  def apply(ir: IR, controlFlowPreventsSplit: Memo[Unit]): Memo[Unit] = {
    val m = Memo.empty[Unit]

    val splitThreshold = HailContext.getFlag("method_split_ir_limit").toInt
    require(splitThreshold > 0, s"invalid method_split_ir_limit")

    def recurAndComputeSizeUnderneath(x: IR): Int = {
      val sizeUnderneath = x.children.iterator.map { case child: IR => recurAndComputeSizeUnderneath(child) }.sum

      val shouldSplit = x match {
        case tl: TailLoop =>
          !controlFlowPreventsSplit.contains(tl) // split if not in a nested loop
        case x if sizeUnderneath > splitThreshold && !controlFlowPreventsSplit.contains(x) => true
        case _ => false
      }
      if (shouldSplit) {
        m.bind(x, ())
        0 // method call is small
      } else {
        sizeUnderneath + (x match {
          case _: Ref => 0
          case _: In => 0
          case _ if IsConstant(x) => 0
          case _ => 1
        })
      }
    }
    recurAndComputeSizeUnderneath(ir)
    m
  }
}
