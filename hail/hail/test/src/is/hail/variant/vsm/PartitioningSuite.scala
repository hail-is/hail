package is.hail.variant.vsm

import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir
import is.hail.expr.ir.{Interpret, MatrixAnnotateRowsTable, TableLiteral, TableValue}
import is.hail.rvd.RVD
import is.hail.types.virtual.{TInt32, TStruct, TableType}

import org.junit.jupiter.api.Test

class PartitioningSuite {
  @Test def testShuffleOnEmptyRDD(implicit ctx: ExecuteContext): Unit = {
    val typ = TableType(TStruct("tidx" -> TInt32), FastSeq("tidx"), TStruct.empty)
    val t = TableLiteral(
      TableValue(
        ctx,
        typ,
        BroadcastRow.empty(ctx),
        RVD.empty(ctx, typ.canonicalRVDType),
      ),
      ctx.theHailClassLoader,
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
