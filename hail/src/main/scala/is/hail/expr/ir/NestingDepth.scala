package is.hail.expr.ir

object NestingDepth {
  def apply(ir0: IR): Memo[Int] = {

    val memo = Memo.empty[Int]

    def memoize(ir: IR, depth: Int): Unit = {
      memo.bind(ir, depth)
      ir match {
        case ArrayMap(a, name, body) =>
          memoize(a, depth)
          memoize(body, depth + 1)
        case ArrayFor(a, valueName, body) =>
          memoize(a, depth)
          memoize(body, depth + 1)
        case ArrayFlatMap(a, name, body) =>
          memoize(a, depth)
          memoize(body, depth + 1)
        case ArrayFilter(a, name, cond) =>
          memoize(a, depth)
          memoize(cond, depth + 1)
        case ArrayFold(a, zero, accumName, valueName, body) =>
          memoize(a, depth)
          memoize(zero, depth)
          memoize(body, depth + 1)
        case ArrayScan(a, zero, accumName, valueName, body) =>
          memoize(a, depth)
          memoize(zero, depth)
          memoize(body, depth + 1)
        case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
          memoize(left, depth)
          memoize(right, depth)
          memoize(keyF, depth + 1)
          memoize(joinF, depth + 1)
        case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) =>
          constructorArgs.foreach(memoize(_, depth))
          initOpArgs.foreach(_.foreach(memoize(_, depth)))
          seqOpArgs.foreach(memoize(_, 0))
        case _ =>
          Children(ir).foreach {
            case child: IR => memoize(child, depth)
            case _ =>
          }
      }
    }
    memoize(ir0, 0)
    memo
  }
}
