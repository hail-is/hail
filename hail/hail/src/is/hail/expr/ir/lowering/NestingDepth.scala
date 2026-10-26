package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{BaseIR, IR, Memo, RefEquality, Scope, UsesAggEnv, UsesScanEnv}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.invariant.NoSharedNodes
import is.hail.utils.TimedBlock

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

  def lookupBinding(x: Block, scope: Scope): Int = scope match {
    case Scope.EVAL => memo(x).eval
    case Scope.AGG => memo(x).agg
    case Scope.SCAN => memo(x).scan
  }
}

object NestingDepth {
  def apply(ctx: ExecuteContext, ir0: BaseIR): NestingDepth =
    TimedBlock.enter {

      val memo = Memo.empty[ScopedDepth]

      def computeChildren(ir: BaseIR): Unit =
        ir.children
          .zipWithIndex
          .foreach {
            case (child: IR, _) => computeIR(child, ScopedDepth(0, 0, 0))
            case (ir: BaseIR, _) => computeChildren(ir)
          }

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
            computeIR(body, ScopedDepth(depth.eval + 1, depth.agg, depth.eval + 1))
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
            computeIR(aggBody, depth.incrementEval.incrementScanOrAgg(isScan)) // per element
            knownLength.foreach(computeIR(_, depth))
          case AggGroupBy(key, aggIR, isScan) =>
            computeIR(key, depth.promoteScanOrAgg(isScan))
            computeIR(aggIR, depth.incrementEval) // per group
          case AggFold(zero, seqOp, combOp, _, _, isScan) =>
            computeIR(zero, depth)
            computeIR(seqOp, depth.promoteScanOrAgg(isScan))
            computeIR(combOp, depth.incrementEval) // per merge
          case ArraySort(a, _, _, lessThan) =>
            computeIR(a, depth)
            computeIR(lessThan, depth.incrementEval)
          case ArrayMaximalIndependentSet(edges, tieBreaker) =>
            computeIR(edges, depth)
            tieBreaker.foreach { case (_, _, tb) => computeIR(tb, depth.incrementEval) }
          case RunAggScan(a, _, init, seqs, result, _) =>
            computeIR(a, depth)
            computeIR(init, depth)
            computeIR(seqs, depth.incrementEval)
            computeIR(result, depth.incrementEval)
          case StreamBufferedAggregate(stream, initAggs, newKey, seqOps, _, _, _) =>
            computeIR(stream, depth)
            FastSeq(initAggs, newKey, seqOps).foreach(computeIR(_, depth.incrementEval))
          // a fresh scope that runs once per partition; only RelationalRefs see this depth
          case CollectDistributedArray(ctxs, globals, _, _, body, dynamicID, _, _) =>
            computeIR(ctxs, depth)
            computeIR(globals, depth)
            computeIR(body, depth.incrementEval)
            computeIR(dynamicID, depth)
          case TableAggregate(child, query) =>
            computeChildren(child)
            computeIR(query, ScopedDepth(0, 0, 0))
          case MatrixAggregate(child, query) =>
            computeChildren(child)
            computeIR(query, ScopedDepth(0, 0, 0))
          case _ =>
            ir.children
              .zipWithIndex
              .foreach {
                case (child: IR, i) =>
                  if (UsesAggEnv(ir, i)) computeIR(child, depth.promoteAgg)
                  else if (UsesScanEnv(ir, i)) computeIR(child, depth.promoteScan)
                  else computeIR(child, depth)
                case (child: BaseIR, _) => computeChildren(child)
              }
        }
      }

      NoSharedNodes.verify(ctx, ir0)

      ir0 match {
        case ir: IR => computeIR(ir, ScopedDepth(0, 0, 0))
        case ir: BaseIR => computeChildren(ir)
      }

      new NestingDepth(memo)
    }
}
