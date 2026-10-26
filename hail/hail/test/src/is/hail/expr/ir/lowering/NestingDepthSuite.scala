package is.hail.expr.ir.lowering

import is.hail.ParameterizedTest
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

import scala.collection.immutable.ArraySeq

class NestingDepthSuite {

  def range: IR = StreamRange(I32(0), I32(10), I32(1))

  // Each child can run more than once each time its parent runs, so a `Ref` in it is one
  // deeper than the parent. NestingDepth reads only the tree's shape, so these needn't type
  // check.

  def testChildRunsRepeatedly(): IndexedSeq[(Ref, IR)] = {

    type P = IndexedSeq[Ref => (Ref, IR)]

    val isScanCases: P =
      FastSeq(false, true).flatMap { isScan =>
        ArraySeq(
          x => x -> aggArrayPerElement(ToArray(range), isScan = isScan)((_, _) => x),
          x => x -> AggGroupBy(I32(0), x, isScan),
          // the comb drops the agg scope, so it isn't promoted to the agg depth either
          x => x -> aggFoldIR(I32(0), isScan)(acc => acc)((_, _) => x),
        )
      }

    val others: P =
      ArraySeq(
        x => x -> StreamAggScan(range, freshName(), x),
        x => x -> StreamAggScan(range, freshName(), ApplyScanOp(Collect())(x)),
        x => x -> ArraySort(range, freshName(), freshName(), x),
        x => x -> RunAggScan(range, freshName(), Void(), x, Void(), FastSeq()),
        x => x -> RunAggScan(range, freshName(), Void(), Void(), x, FastSeq()),
        x => x -> StreamBufferedAggregate(range, x, Void(), Void(), freshName(), FastSeq(), 8),
        x => x -> StreamBufferedAggregate(range, Void(), x, Void(), freshName(), FastSeq(), 8),
        x => x -> StreamBufferedAggregate(range, Void(), Void(), x, freshName(), FastSeq(), 8),
        x => x -> cdaIR(range, makestruct(), "test")((_, _) => x),
        x => x -> ArrayMaximalIndependentSet(range, Some((freshName(), freshName(), x))),
      )

    // NestingDepth requires NoSharedNames
    // Use a fresh Ref x for each test - the alternative would invalidate the
    // invariant because of shared state between tests.
    (isScanCases ++ others).map(_(Ref(Name("x"), TInt32)))
  }

  @ParameterizedTest
  def testChildRunsRepeatedly(x: Ref, ir: IR)(implicit ctx: ExecuteContext): Unit =
    assert(NestingDepth(ctx, ir).lookupRef(x) == 1)
}
