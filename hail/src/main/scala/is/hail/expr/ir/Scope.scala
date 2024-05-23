package is.hail.expr.ir

object UsesAggEnv {
  def apply(ir0: BaseIR, i: Int): Boolean = ir0 match {
    case AggLet(_, _, _, false) => i == 0
    case AggGroupBy(_, _, false) => i == 0
    case AggFilter(_, _, false) => i == 0
    case AggExplode(_, _, _, false) => i == 0
    case AggArrayPerElement(_, _, _, _, _, false) => i == 0
    case ApplyAggOp(initOp, _, _) => i >= initOp.length
    case AggFold(_, _, _, _, _, false) => i > 0
    case _ => false
  }
}

object UsesScanEnv {
  def apply(ir0: BaseIR, i: Int): Boolean = ir0 match {
    case AggLet(_, _, _, true) => i == 0
    case AggGroupBy(_, _, true) => i == 0
    case AggFilter(_, _, true) => i == 0
    case AggExplode(_, _, _, true) => i == 0
    case AggArrayPerElement(_, _, _, _, _, true) => i == 0
    case ApplyScanOp(initOp, _, _) => i >= initOp.length
    case AggFold(_, _, _, _, _, true) => i > 0
    case _ => false
  }
}

object Scope {
  val EVAL: Int = 0
  val AGG: Int = 1
  val SCAN: Int = 2

  def apply(ir0: BaseIR): Memo[Int] = {
    val memo = Memo.empty[Int]

    def compute(ir: BaseIR, scope: Int): Unit = {
      if (ir.isInstanceOf[IR])
        memo.bind(ir, scope)

      ir match {
        case TableAggregate(child, query) =>
          compute(child, EVAL)
          compute(query, EVAL)
        case MatrixAggregate(child, query) =>
          compute(child, EVAL)
          compute(query, EVAL)
        case RelationalLet(_, value, body) =>
          compute(value, EVAL)
          compute(body, scope)
        case _ => ir.children.zipWithIndex.foreach {
            case (child: IR, i) =>
              val usesAgg = UsesAggEnv(ir, i)
              val usesScan = UsesScanEnv(ir, i)
              if (usesAgg) {
                assert(!usesScan)
                assert(scope == EVAL)
                compute(child, AGG)
              } else if (usesScan) {
                assert(scope == EVAL)
                compute(child, SCAN)
              } else
                compute(child, scope)
            case (child, _) => compute(child, EVAL)
          }
      }
    }

    compute(ir0, EVAL)
    memo
  }
}
