package is.hail.expr.ir

object InAgg {

  def apply(ir0: IR): Memo[Boolean] = {
    val memo = Memo.empty[Boolean]

    def memoize(ir: IR, inAgg: Boolean): Unit = {
      memo.bind(ir, inAgg)

      ir match {
        case AggGroupBy(key, aggIR) =>
          assert(!inAgg)
          memoize(key, inAgg = true)
          memoize(aggIR, inAgg = false)
        case AggFilter(cond, aggIR) =>
          assert(!inAgg)
          memoize(cond, inAgg = true)
          memoize(aggIR, inAgg = false)
        case AggExplode(a, _, aggIR) =>
          assert(!inAgg)
          memoize(a, inAgg = true)
          memoize(aggIR, inAgg = false)
        case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
          assert(!inAgg)
          constructorArgs.foreach(memoize(_, inAgg = false))
          initOpArgs.foreach(_.foreach(memoize(_, inAgg = false)))
          seqOpArgs.foreach(memoize(_, inAgg = true))
        case AggArrayPerElement(a, name, aggBody) =>
          assert(!inAgg)
          memoize(a, inAgg = true)
          memoize(aggBody, inAgg = false)
        case _ => VisitIR(memoize(_, inAgg))(ir)
      }
    }
    memoize(ir0, inAgg = false)
    memo
  }
}
