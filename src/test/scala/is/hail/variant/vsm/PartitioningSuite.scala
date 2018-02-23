package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.check.{Gen, Prop}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.variant.{GenomeReference, MatrixTable, VSMSubgen}
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
      GenomeReference.addReference(vds.genomeReference)

      val out = tmpDir.createTempFile("out", ".vds")
      val out2 = tmpDir.createTempFile("out", ".vds")

      vds.typecheck()
      val orig = vds.coalesce(nPar)
      orig.write(out)
      val problem = hc.readVDS(out)

      hc.readVDS(out).annotateVariantsExpr("va = va").countVariants()

      // need to do 2 writes to ensure that the RDD is ordered
      hc.readVDS(out)
        .write(out2)

      val readback = hc.readVDS(out2)

      val result = compare(orig, readback)

      GenomeReference.removeReference(vds.genomeReference.name)
      result
    }.check()
  }

  @Test def testHintPartitionerAdjustedCorrectly() {
    val mt = MatrixTable.fromRowsTable(Table.range(hc, 100, "idx", partitions=Some(6)))

    val trows = (for (i <- -5 to 200) yield { Row(i, "foo") }).reverse
    val t = Table.parallelize(hc, trows, TStruct("tidx"-> TInt32(), "foo"-> TString()), Array("tidx"), nPartitions=Some(6))

    mt.annotateVariantsTable(t, "foo").forceCountRows()
  }
}
