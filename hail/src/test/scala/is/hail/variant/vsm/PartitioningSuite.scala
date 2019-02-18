package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir
import is.hail.expr.ir.{Interpret, MatrixAnnotateRowsTable, TableLiteral, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TInt32, TStruct}
import is.hail.rvd.RVD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PartitioningSuite extends SparkSuite {
  @Test def testShuffleOnEmptyRDD() {
    val typ = TableType(TStruct("tidx" -> TInt32()), IndexedSeq("tidx"), TStruct.empty())
    val t = TableLiteral(TableValue(
      typ, BroadcastRow(Row.empty, TStruct.empty(), sc), RVD.empty(sc, typ.canonicalRVDType)))
    val rangeReader = ir.MatrixRangeReader(100, 10, Some(10))
    Interpret(
      MatrixAnnotateRowsTable(
        ir.MatrixRead(rangeReader.fullType, false, false, rangeReader),
        t,
        "foo"))
      .rvd.count()
  }
}
