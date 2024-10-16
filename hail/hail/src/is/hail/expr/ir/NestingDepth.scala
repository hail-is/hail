package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._

case class ScopedDepth(eval: Int, agg: Int, scan: Int) {
  def incrementEval: ScopedDepth = ScopedDepth(eval + 1, agg, scan)

  def incrementAgg: ScopedDepth = ScopedDepth(eval, agg + 1, scan)

  def promoteAgg: ScopedDepth = ScopedDepth(agg, 0, 0)

  def incrementScan: ScopedDepth = ScopedDepth(eval, agg, scan + 1)

  def promoteScan: ScopedDepth = ScopedDepth(scan, 0, 0)

  def incrementScanOrAgg(isScan: Boolean): ScopedDepth = if (isScan) incrementScan else incrementAgg

  def promoteScanOrAgg(isScan: Boolean): ScopedDepth = if (isScan) promoteScan else promoteAgg
}

final class NestingDepth(private val memo: Memo[ScopedDepth]) {
  def lookupRef(x: RefEquality[BaseRef]): Int = memo.lookup(x).eval
  def lookupRef(x: BaseRef): Int = memo.lookup(x).eval

  def lookupBinding(x: Block, scope: Int): Int = scope match {
    case Scope.EVAL => memo(x).eval
    case Scope.AGG => memo(x).agg
    case Scope.SCAN => memo(x).scan
  }
}

object NestingDepth {
  def apply(ctx: ExecuteContext, ir0: BaseIR): NestingDepth =
    ctx.time {

      val memo = Memo.empty[ScopedDepth]

      def computeChildren(ir: BaseIR): Unit = {
        ir.children
          .zipWithIndex
          .foreach {
            case (child: IR, _) => computeIR(child, ScopedDepth(0, 0, 0))
            case (tir: TableIR, _) => computeTable(tir)
            case (mir: MatrixIR, _) => computeMatrix(mir)
            case (bmir: BlockMatrixIR, _) => computeBlockMatrix(bmir)
          }
      }

      def computeTable(tir: TableIR): Unit = computeChildren(tir)

      def computeMatrix(mir: MatrixIR): Unit = computeChildren(mir)

      def computeBlockMatrix(bmir: BlockMatrixIR): Unit = computeChildren(bmir)

      def computeIR(ir: IR, depth: ScopedDepth): Unit = {
        ir match {
          case _: Block | _: BaseRef =>
            memo.bind(ir, depth)
          case _ =>
        }
        ir match {
          case StreamMap(a, _, body) =>
            computeIR(a, depth)
            computeIR(body, depth.incrementEval)
          case StreamAgg(a, _, body) =>
            computeIR(a, depth)
            computeIR(body, ScopedDepth(depth.eval, depth.eval + 1, depth.scan))
          case StreamAggScan(a, _, body) =>
            computeIR(a, depth)
            computeIR(body, ScopedDepth(depth.eval, depth.agg, depth.eval + 1))
          case StreamZip(as, _, body, _, _) =>
            as.foreach(computeIR(_, depth))
            computeIR(body, depth.incrementEval)
          case StreamZipJoin(as, _, _, _, joinF) =>
            as.foreach(computeIR(_, depth))
            computeIR(joinF, depth.incrementEval)
          case StreamZipJoinProducers(contexts, _, makeProducer, _, _, _, joinF) =>
            computeIR(contexts, depth)
            computeIR(makeProducer, depth.incrementEval)
            computeIR(joinF, depth.incrementEval)
          case StreamFor(a, _, body) =>
            computeIR(a, depth)
            computeIR(body, depth.incrementEval)
          case StreamFlatMap(a, _, body) =>
            computeIR(a, depth)
            computeIR(body, depth.incrementEval)
          case StreamFilter(a, _, cond) =>
            computeIR(a, depth)
            computeIR(cond, depth.incrementEval)
          case StreamTakeWhile(a, _, cond) =>
            computeIR(a, depth)
            computeIR(cond, depth.incrementEval)
          case StreamDropWhile(a, _, cond) =>
            computeIR(a, depth)
            computeIR(cond, depth.incrementEval)
          case StreamFold(a, zero, _, _, body) =>
            computeIR(a, depth)
            computeIR(zero, depth)
            computeIR(body, depth.incrementEval)
          case StreamFold2(a, accum, _, seq, result) =>
            computeIR(a, depth)
            accum.foreach { case (_, value) => computeIR(value, depth) }
            seq.foreach(computeIR(_, depth.incrementEval))
            computeIR(result, depth)
          case StreamScan(a, zero, _, _, body) =>
            computeIR(a, depth)
            computeIR(zero, depth)
            computeIR(body, depth.incrementEval)
          case StreamJoinRightDistinct(left, right, _, _, _, _, joinF, _) =>
            computeIR(left, depth)
            computeIR(right, depth)
            computeIR(joinF, depth.incrementEval)
          case StreamLeftIntervalJoin(left, right, _, _, _, _, body) =>
            computeIR(left, depth)
            computeIR(right, depth)
            computeIR(body, depth.incrementEval)
          case TailLoop(_, params, _, body) =>
            params.foreach { case (_, p) => computeIR(p, depth) }
            computeIR(body, depth.incrementEval)
          case NDArrayMap(nd, _, body) =>
            computeIR(nd, depth)
            computeIR(body, depth.incrementEval)
          case NDArrayMap2(nd1, nd2, _, _, body, _) =>
            computeIR(nd1, depth)
            computeIR(nd2, depth)
            computeIR(body, depth.incrementEval)
          case AggExplode(array, _, aggBody, isScan) =>
            computeIR(array, depth.promoteScanOrAgg(isScan))
            computeIR(aggBody, depth.incrementScanOrAgg(isScan))
          case AggArrayPerElement(a, _, _, aggBody, knownLength, isScan) =>
            computeIR(a, depth.promoteScanOrAgg(isScan))
            computeIR(aggBody, depth.incrementScanOrAgg(isScan))
            knownLength.foreach(computeIR(_, depth))
          case TableAggregate(child, query) =>
            computeTable(child)
            computeIR(query, ScopedDepth(0, 0, 0))
          case MatrixAggregate(child, query) =>
            computeMatrix(child)
            computeIR(query, ScopedDepth(0, 0, 0))
          case _ =>
            ir.children
              .zipWithIndex
              .foreach {
                case (child: IR, i) => if (UsesAggEnv(ir, i))
                    computeIR(child, depth.promoteAgg)
                  else if (UsesScanEnv(ir, i))
                    computeIR(child, depth.promoteScan)
                  else
                    computeIR(child, depth)
                case (child: TableIR, _) => computeTable(child)
                case (child: MatrixIR, _) => computeMatrix(child)
                case (child: BlockMatrixIR, _) => computeBlockMatrix(child)
              }
        }
      }

      ir0 match {
        case ir: IR => computeIR(ir, ScopedDepth(0, 0, 0))
        case tir: TableIR => computeTable(tir)
        case mir: MatrixIR => computeMatrix(mir)
        case bmir: BlockMatrixIR => computeBlockMatrix(bmir)
      }

      new NestingDepth(memo)
    }
}
