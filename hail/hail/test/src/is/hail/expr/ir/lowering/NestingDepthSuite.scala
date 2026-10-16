package is.hail.expr.ir.lowering

import is.hail.ParameterizedTest
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

class NestingDepthSuite {

  def isScan = FastSeq(false, true)

  // AggFold's comb drops the agg scope (Bindings.get), so it isn't nested in StreamAgg's loop
  @ParameterizedTest("isScan")
  def testAggFoldCombNotPromoted(isScan: Boolean)(implicit ctx: ExecuteContext): Unit = {
    val t = Ref(freshName(), TInt32)
    var block: Block = null

    def fold(x: Atom): IR =
      aggFoldIR(I32(0), isScan)(_ + x) { (l, r) =>
        block = Let(FastSeq(t.name -> (l + r)), t).asInstanceOf[Block]
        block
      }

    val s = StreamRange(I32(0), I32(10), I32(1))
    val ir = if (isScan) s.streamAggScan(fold) else s.streamAgg(fold)
    assert(NestingDepth(ctx, ir).lookupBinding(block, Scope.EVAL) == 0)
  }
}
