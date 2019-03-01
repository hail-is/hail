package is.hail.expr.ir

object InAgg {

  def apply(ir0: BaseIR): Memo[Boolean] = {
    val memo = Memo.empty[Boolean]

    def compute(ir: BaseIR, inAgg: Boolean): Unit = {
      memo.bind(ir, inAgg)

      ir match {
        case AggLet(_, value, body) =>
          assert(!inAgg)
          compute(value, inAgg = true)
          compute(body, inAgg = false)
        case AggGroupBy(key, aggIR) =>
          assert(!inAgg)
          compute(key, inAgg = true)
          compute(aggIR, inAgg = false)
        case AggFilter(cond, aggIR) =>
          assert(!inAgg)
          compute(cond, inAgg = true)
          compute(aggIR, inAgg = false)
        case AggExplode(a, _, aggIR) =>
          assert(!inAgg)
          compute(a, inAgg = true)
          compute(aggIR, inAgg = false)
        case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, _) =>
          assert(!inAgg)
          constructorArgs.foreach(compute(_, inAgg = false))
          initOpArgs.foreach(_.foreach(compute(_, inAgg = false)))
          seqOpArgs.foreach(compute(_, inAgg = true))
        case AggArrayPerElement(a, _, aggBody) =>
          assert(!inAgg)
          compute(a, inAgg = true)
          compute(aggBody, inAgg = false)
        case TableAggregate(child, query) =>
          compute(child, false)
          compute(query, false)
        case MatrixAggregate(child, query) =>
          compute(child, false)
          compute(query, false)
        case _ => ir.children.foreach {
          case vir: IR => compute(vir, inAgg)
          case child => compute(child, false)
        }
      }
    }
    compute(ir0, inAgg = false)
    memo
  }
}
