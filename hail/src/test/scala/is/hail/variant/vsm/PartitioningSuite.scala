package is.hail.variant.vsm

import is.hail.HailSuite
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir
import is.hail.expr.ir.{ExecuteContext, Interpret, MatrixAnnotateRowsTable, TableLiteral, TableRange, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TInt32, TStruct}
import is.hail.rvd.RVD
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.Test

class PartitioningSuite extends HailSuite {
  @Test def testShuffleOnEmptyRDD() {
    val typ = TableType(TStruct("tidx" -> TInt32), FastIndexedSeq("tidx"), TStruct.empty)
    val t = TableLiteral(TableValue(
      typ, BroadcastRow.empty(ctx), RVD.empty(sc, typ.canonicalRVDType)), ctx)
    val rangeReader = ir.MatrixRangeReader(100, 10, Some(10))
    Interpret(
      MatrixAnnotateRowsTable(
        ir.MatrixRead(rangeReader.fullMatrixType, false, false, rangeReader),
        t,
        "foo",
        product = false),
      ctx, optimize = false)
      .rvd.count()
  }

  @Test def testEmptyRDDOrderedJoin() {
    val tv = Interpret.apply(TableRange(100, 6), ctx)

    val nonEmptyRVD = tv.rvd
    val rvdType = nonEmptyRVD.typ
    val emptyRVD = RVD.empty(hc.sc, rvdType)

    ExecuteContext.scoped { ctx =>
      emptyRVD.orderedJoin(nonEmptyRVD, "left", (_, it) => it.map(_._1), rvdType, ctx).count()
      emptyRVD.orderedJoin(nonEmptyRVD, "inner", (_, it) => it.map(_._1), rvdType, ctx).count()
      nonEmptyRVD.orderedJoin(emptyRVD, "left", (_, it) => it.map(_._1), rvdType, ctx).count()
      nonEmptyRVD.orderedJoin(emptyRVD, "inner", (_, it) => it.map(_._1), rvdType, ctx).count()
    }
  }
}
