package is.hail.expr.ir

object NestingDepth {
  def apply(ir0: BaseIR): Memo[Int] = {

    val memo = Memo.empty[Int]

    def computeChildren(ir: BaseIR, depth: Option[Int] = None): Unit = {
      ir.children
        .foreach {
          case child: IR => computeIR(child, depth.getOrElse(0))
          case tir: TableIR => computeTable(tir)
          case mir: MatrixIR => computeMatrix(mir)
          case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
        }
    }

    def computeTable(tir: TableIR): Unit = computeChildren(tir)

    def computeMatrix(mir: MatrixIR): Unit = computeChildren(mir)

    def computeBlockMatrix(bmir: BlockMatrixIR): Unit = computeChildren(bmir)

    def computeIR(ir: IR, depth: Int): Unit = {
      memo.bind(ir, depth)
      ir match {
        case ArrayMap(a, name, body) =>
          computeIR(a, depth)
          computeIR(body, depth + 1)
        case ArrayFor(a, valueName, body) =>
          computeIR(a, depth)
          computeIR(body, depth + 1)
        case ArrayFlatMap(a, name, body) =>
          computeIR(a, depth)
          computeIR(body, depth + 1)
        case ArrayFilter(a, name, cond) =>
          computeIR(a, depth)
          computeIR(cond, depth + 1)
        case ArrayFold(a, zero, accumName, valueName, body) =>
          computeIR(a, depth)
          computeIR(zero, depth)
          computeIR(body, depth + 1)
        case ArrayScan(a, zero, accumName, valueName, body) =>
          computeIR(a, depth)
          computeIR(zero, depth)
          computeIR(body, depth + 1)
        case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
          computeIR(left, depth)
          computeIR(right, depth)
          computeIR(keyF, depth + 1)
          computeIR(joinF, depth + 1)
        case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, _) =>
          constructorArgs.foreach(computeIR(_, 0))
          initOpArgs.foreach(_.foreach(computeIR(_, depth)))
          seqOpArgs.foreach(computeIR(_, 0))
        case _ =>
          computeChildren(ir, Some(depth))
      }
    }

    ir0 match {
      case ir: IR => computeIR(ir, 0)
      case tir: TableIR => computeTable(tir)
      case mir: MatrixIR => computeMatrix(mir)
      case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
    }

    memo
  }
}
