package is.hail.expr.ir

object UsesAggEnv {
  def apply(ir0: BaseIR, i: Int): Boolean = ir0 match {
    case _: AggLet => i == 1
    case _: AggGroupBy => i == 0
    case _: AggFilter => i == 0
    case _: AggExplode => i == 0
    case ApplyAggOp(ctor, initOp, _, _) => i >= ctor.length + initOp.map(_.length).getOrElse(0)
    case _: AggArrayPerElement => i == 0
    case _ => false
  }
}


object InAgg {

  def apply(ir0: BaseIR): Memo[Boolean] = {
    val memo = Memo.empty[Boolean]

    def compute(ir: BaseIR, inAgg: Boolean): Unit = {
      memo.bind(ir, inAgg)

      ir match {
        case TableAggregate(child, query) =>
          compute(child, false)
          compute(query, false)
        case MatrixAggregate(child, query) =>
          compute(child, false)
          compute(query, false)
        case _ => ir.children.iterator.zipWithIndex.foreach {
          case (child: IR, i) =>
            val usesAgg = UsesAggEnv(ir, i)
            if (usesAgg)
              assert(!inAgg)
            compute(child, inAgg || usesAgg)
          case (child, i) => compute(child, false)
        }
      }
    }
    compute(ir0, inAgg = false)
    memo
  }
}
