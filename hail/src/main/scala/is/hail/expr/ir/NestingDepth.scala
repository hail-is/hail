package is.hail.expr.ir

object NestingDepth {
  def apply(ir0: IR): Memo[Int] = {

    val memo = Memo.empty[Int]

    def compute(ir: IR, depth: Int): Unit = {
      memo.bind(ir, depth)
      ir match {
        case ArrayMap(a, name, body) =>
          compute(a, depth)
          compute(body, depth + 1)
        case ArrayFor(a, valueName, body) =>
          compute(a, depth)
          compute(body, depth + 1)
        case ArrayFlatMap(a, name, body) =>
          compute(a, depth)
          compute(body, depth + 1)
        case ArrayFilter(a, name, cond) =>
          compute(a, depth)
          compute(cond, depth + 1)
        case ArrayFold(a, zero, accumName, valueName, body) =>
          compute(a, depth)
          compute(zero, depth)
          compute(body, depth + 1)
        case ArrayScan(a, zero, accumName, valueName, body) =>
          compute(a, depth)
          compute(zero, depth)
          compute(body, depth + 1)
        case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
          compute(left, depth)
          compute(right, depth)
          compute(keyF, depth + 1)
          compute(joinF, depth + 1)
        case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
          constructorArgs.foreach(compute(_, depth))
          initOpArgs.foreach(_.foreach(compute(_, depth)))
          seqOpArgs.foreach(compute(_, 0))
        case _ =>
          Children(ir).foreach {
            case child: IR => compute(child, depth)
            case _ =>
          }
      }
    }
    compute(ir0, 0)
    memo
  }
}
