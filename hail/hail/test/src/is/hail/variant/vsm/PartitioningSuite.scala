package is.hail.variant.vsm

import is.hail.HailSuite
import is.hail.annotations.BroadcastRow
import is.hail.collection.FastSeq
import is.hail.expr.ir
import is.hail.expr.ir.{Interpret, MatrixAnnotateRowsTable, TableLiteral, TableValue}
import is.hail.rvd.RVD
import is.hail.types.virtual.{TInt32, TStruct, TableType}

import org.testng.annotations.Test

class PartitioningSuite extends HailSuite {
  @Test def testShuffleOnEmptyRDD(): Unit = {
    val typ = TableType(TStruct("tidx" -> TInt32), FastSeq("tidx"), TStruct.empty)
    val t = TableLiteral(
      TableValue(
        ctx,
        typ,
        BroadcastRow.empty(ctx),
        RVD.empty(ctx, typ.canonicalRVDType),
      ),
      theHailClassLoader,
    )
    val rangeReader = ir.MatrixRangeReader(ctx, 100, 10, Some(10))
    unoptimized { ctx =>
      Interpret(
        MatrixAnnotateRowsTable(
          ir.MatrixRead(rangeReader.fullMatrixType, false, false, rangeReader),
          t,
          "foo",
          product = false,
        ),
        ctx,
      )
        .rvd.count(): Unit
    }
  }
}
