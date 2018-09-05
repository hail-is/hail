package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.annotations.BroadcastRow
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir
import is.hail.expr.ir.{MatrixAnnotateRowsTable, TableLiteral, TableValue}
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, UnpartitionedRVD}
import is.hail.table.Table
import is.hail.variant.{MatrixTable, ReferenceGenome, VSMSubgen}
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
    val typ = TableType(TStruct("tidx" -> TInt32()), Some(IndexedSeq("tidx")), TStruct.empty())
    val t = TableLiteral(TableValue(
      typ, BroadcastRow(Row.empty, TStruct.empty(), sc), OrderedRVD.empty(sc, typ.rvdType)))
    val rangeReader = ir.MatrixRangeReader(100, 10, Some(10))
    MatrixAnnotateRowsTable(ir.MatrixRead(rangeReader.fullType, false, false, rangeReader), t, "foo", None)
      .execute(hc).rvd.count()
  }

  @Test def testEmptyRightRDDOrderedJoinDistinct() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, nPartitions = Some(6)))
    val orvdType = mt.matrixType.orvdType

    mt.rvd.orderedJoinDistinct(OrderedRVD.empty(hc.sc, orvdType), "left", (_, it) => it.map(_._1), orvdType).count()
    mt.rvd.orderedJoinDistinct(OrderedRVD.empty(hc.sc, orvdType), "inner", (_, it) => it.map(_._1), orvdType).count()
  }

  @Test def testEmptyRDDOrderedJoin() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, nPartitions = Some(6)))
    val orvdType = mt.matrixType.orvdType

    val nonEmptyRVD = mt.rvd
    val emptyRVD = OrderedRVD.empty(hc.sc, orvdType)

    emptyRVD.orderedJoin(nonEmptyRVD, "left", (_, it) => it.map(_._1), orvdType).count()
    emptyRVD.orderedJoin(nonEmptyRVD, "inner", (_, it) => it.map(_._1), orvdType).count()
    nonEmptyRVD.orderedJoin(emptyRVD, "left", (_, it) => it.map(_._1), orvdType).count()
    nonEmptyRVD.orderedJoin(emptyRVD, "inner", (_, it) => it.map(_._1), orvdType).count()
  }
}
