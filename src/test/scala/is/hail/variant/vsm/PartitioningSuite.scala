package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.annotations.{Annotation, BroadcastValue}
import is.hail.check.{Gen, Prop}
import is.hail.expr.{TableLiteral, TableValue}
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

  @Test def testWriteRead() {
    Prop.forAll(MatrixTable.gen(hc, VSMSubgen.random), Gen.choose(1, 10)) { case (vds, nPar) =>
      ReferenceGenome.addReference(vds.referenceGenome)

      val out = tmpDir.createTempFile("out", ".vds")
      val out2 = tmpDir.createTempFile("out", ".vds")

      vds.typecheck()
      val orig = vds.coalesce(nPar)
      orig.write(out)
      val problem = hc.readVDS(out)

      hc.readVDS(out).annotateRowsExpr("va = va").countRows()

      // need to do 2 writes to ensure that the RDD is ordered
      hc.readVDS(out)
        .write(out2)

      val readback = hc.readVDS(out2)

      val result = compare(orig, readback)

      ReferenceGenome.removeReference(vds.referenceGenome.name)
      result
    }.check()
  }

  @Test def testHintPartitionerAdjustedCorrectly() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, "idx", partitions=Some(6)))
    val t = Table.range(hc, 205, "idx", partitions=Some(6))
      .select("tidx = 200 - row.idx")
      .keyBy("tidx")
    mt.annotateRowsTable(t, "foo").forceCountRows()
  }

  @Test def testShuffleOnEmptyRDD() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, "idx", partitions=Some(6)))
    val t = new Table(hc,
      TableLiteral(TableValue(
        TableType(TStruct("tidx"->TInt32()), Array("tidx"), TStruct.empty()),
        BroadcastValue(Annotation.empty, TStruct.empty(), sc),
        UnpartitionedRVD.empty(sc, TStruct("tidx"->TInt32())))))
    mt.annotateRowsTable(t, "foo").forceCountRows()
  }

  @Test def testEmptyRightRDDOrderedJoinDistinct() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, "idx", partitions=Some(6)))
    val orvdType = mt.matrixType.orvdType

    mt.rvd.orderedJoinDistinct(OrderedRVD.empty(hc.sc, orvdType), "left").count()
    mt.rvd.orderedJoinDistinct(OrderedRVD.empty(hc.sc, orvdType), "inner").count()
  }
}
