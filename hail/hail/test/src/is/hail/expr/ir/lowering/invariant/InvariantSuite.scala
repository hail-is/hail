package is.hail.expr.ir.lowering.invariant

import is.hail.ParameterizedTest
import is.hail.TestUtils.intercept
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{Memoized => M, _}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.invariant.Flags.StrictInvariants
import is.hail.types.virtual._

import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class InvariantSuite {

  def nestedLetsRedefineName: IR =
    M.eval {
      val xn = Name("x")
      for {
        x <- Let(FastSeq(xn -> I32(1)), Ref(xn, TInt32))
        y <- Let(FastSeq(xn -> (x + x)), x)
      } yield x + y
    }

  def testUniquelyNamed(implicit ctx: ExecuteContext) =
    FastSeq(
      // — eval —
      // nested Lets bind the same name
      nestedLetsRedefineName,
      // nested StreamMaps bind the same element name
      {
        val n = Name("elt")
        StreamMap(rangeIR(10), n, StreamMap(rangeIR(5), n, Ref(n, TInt32)))
      },
      // TableMapRows binds 'global', Let re-binds it
      TableRange(0, 1).mapRows((global, row) => Let(FastSeq(TableIR.globalName -> global), row)),
      // MatrixMapEntries binds 'global' in eval, Let re-binds it
      {
        val mt = MatrixIR.range(ctx, 2, 2, Some(1))
        MatrixMapEntries(
          mt,
          Let(FastSeq(MatrixIR.globalName -> I32(0)), Ref(MatrixIR.globalName, TInt32)),
        )
      },
      // — agg —
      // StreamAgg creates agg scope, AggExplode re-binds the same name in it
      {
        val n = Name("x")
        StreamAgg(
          rangeIR(10),
          n,
          AggExplode(
            MakeStream(FastSeq(Ref(n, TInt32)), TStream(TInt32)),
            n,
            ApplyAggOp(Count())(),
            isScan = false,
          ),
        )
      },
      // x bound in eval, StreamAgg copies eval into agg, AggExplode re-binds x in agg
      {
        val x = Name("x")
        Block(
          FastSeq(Binding(x, I32(1), Scope.EVAL)),
          StreamAgg(
            rangeIR(10),
            Name("elt"),
            AggExplode(
              MakeStream(FastSeq(I32(0)), TStream(TInt32)),
              x,
              ApplyAggOp(Count())(),
              isScan = false,
            ),
          ),
        )
      },
      // MatrixMapRows binds 'col' in agg (but not eval), AggExplode re-binds it in agg
      {
        val mt = MatrixIR.range(ctx, 2, 2, Some(1))
        MatrixMapRows(
          mt,
          AggExplode(
            MakeStream(FastSeq(I32(0)), TStream(TInt32)),
            MatrixIR.colName,
            ApplyAggOp(Count())(),
            isScan = false,
          ),
        )
      },
      // MatrixMapCols binds 'row' in agg (but not eval), AggExplode re-binds it in agg
      {
        val mt = MatrixIR.range(ctx, 2, 2, Some(1))
        MatrixMapCols(
          mt,
          AggExplode(
            MakeStream(FastSeq(I32(0)), TStream(TInt32)),
            MatrixIR.rowName,
            ApplyAggOp(Count())(),
            isScan = false,
          ),
          None,
        )
      },
      // — scan —
      // TableMapRows has scan scope; StreamAggScan re-binds 'row' in eval (also in scan)
      TableRange(0, 1).mapRows { (_, row) =>
        StreamAggScan(
          rangeIR(5),
          TableIR.rowName,
          ApplyScanOp(Count())(),
        )
      },
      // x bound in eval, StreamAggScan copies eval into scan, AggExplode re-binds x in scan
      {
        val x = Name("x")
        Block(
          FastSeq(Binding(x, I32(1), Scope.EVAL)),
          StreamAggScan(
            rangeIR(10),
            Name("elt"),
            AggExplode(
              MakeStream(FastSeq(I32(0)), TStream(TInt32)),
              x,
              ApplyScanOp(Count())(),
              isScan = true,
            ),
          ),
        )
      },
      // — relational —
      // nested RelationalLets bind the same name
      {
        val n = Name("rel")
        RelationalLet(n, I32(1), RelationalLet(n, I32(2), RelationalRef(n, TInt32)))
      },
    )

  @ParameterizedTest def testUniquelyNamed(ir: BaseIR)(implicit ctx: ExecuteContext): Unit =
    assertThrows(
      classOf[UnsatisfiedInvariantError],
      () => UniquelyNamed.verify(ctx, ir),
    )

  @Test def testNoSharedNodes(implicit ctx: ExecuteContext): Unit = {
    val r = Ref(freshName(), TInt32)
    val ir1 = MakeTuple.ordered(FastSeq(I64(1), r, r, I32(1)))
    intercept[UnsatisfiedInvariantError](NoSharedNodes.verify(ctx, ir1)): Unit
    NoSharedNodes.verify(ctx, ir1.deepCopy)
  }

  @Test def testNoOpWithoutStrictInvariants(implicit ctx: ExecuteContext): Unit =
    ctx.local(flags = ctx.flags - StrictInvariants) { ctx =>
      NoSharedNodes.verify(ctx, nestedLetsRedefineName)
    }
}
