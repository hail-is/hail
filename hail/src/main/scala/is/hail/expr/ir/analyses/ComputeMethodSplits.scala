package is.hail.expr.ir.analyses

import is.hail.expr.ir._

object ComputeMethodSplits {
  val splitThreshold: Int = 16

  def apply(ir: IR, controlFlowPreventsSplit: Memo[Unit]): Memo[Unit] = {
    val m = Memo.empty[Unit]

    def recurAndComputeSizeUnderneath(x: IR): Int = {
      val sizeUnderneath = x.children.iterator.map { case child: IR => recurAndComputeSizeUnderneath(child) }.sum


      val shouldSplit = x match {
        case tl: TailLoop =>
          assert(!controlFlowPreventsSplit.contains(tl))
          true
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
