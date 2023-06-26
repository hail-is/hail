package is.hail.expr.ir.analyses

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._

object ComputeMethodSplits {
  def apply(ctx: ExecuteContext, ir: IR, controlFlowPreventsSplit: Memo[Unit]): Memo[Unit] = {
    val m = Memo.empty[Unit]

    val splitThreshold = ctx.getFlag("method_split_ir_limit").toInt
    require(splitThreshold > 0, s"invalid method_split_ir_limit")

    def recurAndComputeSizeUnderneath(x: IR): Int = {
      val sizeUnderneath = x.childrenSeq.iterator.map { case child: IR => recurAndComputeSizeUnderneath(child) }.sum

      val shouldSplit = !controlFlowPreventsSplit.contains(x) && (x match {
        case _: TailLoop => true

        // stream consumers
        case _: ToArray => true
        case _: ToSet => true
        case _: ToDict => true
        case _: StreamFold => true
        case _: StreamFold2 => true
        case _: StreamLen => true
        case _: StreamFor => true

        case _ => sizeUnderneath > splitThreshold
      })
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
