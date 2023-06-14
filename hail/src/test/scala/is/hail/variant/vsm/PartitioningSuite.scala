package is.hail.variant.vsm

import is.hail.HailSuite
import is.hail.annotations.BroadcastRow
import is.hail.backend.ExecuteContext
import is.hail.expr.ir
import is.hail.expr.ir.lowering.Lower.monadLowerInstanceForLower
import is.hail.expr.ir.lowering.LoweringState
import is.hail.expr.ir.{Interpret, MatrixAnnotateRowsTable, TableLiteral, TableRange, TableValue}
import is.hail.rvd.RVD
import is.hail.types._
import is.hail.types.virtual.{TInt32, TStruct}
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class PartitioningSuite extends HailSuite {
  @Test def testShuffleOnEmptyRDD() {
    val typ = TableType(TStruct("tidx" -> TInt32), FastIndexedSeq("tidx"), TStruct.empty)
    val t = TableLiteral[Execute](
      TableValue(
        typ,
        BroadcastRow.empty[Execute].apply(ctx),
        RVD.empty(ctx, typ.canonicalRVDType)
      )
    ).apply(ctx)

    val rangeReader = ir.MatrixRangeReader(100, 10, Some(10))
    Interpret(
      MatrixAnnotateRowsTable(
        ir.MatrixRead(rangeReader.fullMatrixType, false, false, rangeReader),
        t,
        "foo",
        product = false
      ),
      optimize = false
    ).runA(ctx, LoweringState()).rvd.count()
  }

  @Test def testEmptyRDDOrderedJoin() {
    val tv = Interpret.apply(TableRange(100, 6)).runA(ctx, LoweringState())

    val nonEmptyRVD = tv.rvd
    val rvdType = nonEmptyRVD.typ

    ExecuteContext.scoped() { ctx =>
      val emptyRVD = RVD.empty(ctx, rvdType)
      emptyRVD.orderedJoin(nonEmptyRVD, "left", (_, it) => it.map(_._1), rvdType, ctx).count()
      emptyRVD.orderedJoin(nonEmptyRVD, "inner", (_, it) => it.map(_._1), rvdType, ctx).count()
      nonEmptyRVD.orderedJoin(emptyRVD, "left", (_, it) => it.map(_._1), rvdType, ctx).count()
      nonEmptyRVD.orderedJoin(emptyRVD, "inner", (_, it) => it.map(_._1), rvdType, ctx).count()
    }
  }
}
