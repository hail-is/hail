package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir
import is.hail.expr.ir.{Interpret, MatrixAnnotateRowsTable, TableLiteral, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TInt32, TStruct}
import is.hail.rvd.RVD
import is.hail.table.Table
import is.hail.variant.MatrixTable
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PartitioningSuite extends SparkSuite {

  def compare(vds1: MatrixTable, vds2: MatrixTable): Boolean = {
    val s1 = vds1.variantsAndAnnotations
      .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
      .collect()
      .toSet
    val s2 = vds2.variantsAndAnnotations
      .mapPartitionsWithIndex { case (i, it) => it.zipWithIndex.map(x => (i, x)) }
      .collect()
      .toSet
    s1 == s2
  }

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

  @Test def testEmptyRightRDDOrderedJoinDistinct() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, nPartitions = Some(6)))
    val rvdType = mt.matrixType.canonicalRVDType

    mt.rvd.orderedJoinDistinct(RVD.empty(hc.sc, rvdType), "left", (_, it) => it.map(_._1), rvdType).count()
    mt.rvd.orderedJoinDistinct(RVD.empty(hc.sc, rvdType), "inner", (_, it) => it.map(_._1), rvdType).count()
  }

  @Test def testEmptyRDDOrderedJoin() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, nPartitions = Some(6)))
    val rvdType = mt.matrixType.canonicalRVDType

    val nonEmptyRVD = mt.rvd
    val emptyRVD = RVD.empty(hc.sc, rvdType)

    emptyRVD.orderedJoin(nonEmptyRVD, "left", (_, it) => it.map(_._1), rvdType).count()
    emptyRVD.orderedJoin(nonEmptyRVD, "inner", (_, it) => it.map(_._1), rvdType).count()
    nonEmptyRVD.orderedJoin(emptyRVD, "left", (_, it) => it.map(_._1), rvdType).count()
    nonEmptyRVD.orderedJoin(emptyRVD, "inner", (_, it) => it.map(_._1), rvdType).count()
  }
}
