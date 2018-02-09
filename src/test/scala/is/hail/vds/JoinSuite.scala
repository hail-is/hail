package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations.UnsafeRow
import is.hail.expr.types.{TStruct, TVariant}
import is.hail.table.Table
import is.hail.variant.{GenomeReference, Variant, MatrixTable}
import is.hail.utils._
import is.hail.testUtils._

import scala.language.implicitConversions
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class JoinSuite extends SparkSuite {
  @Test def test() {
    val joined = hc.importVCF("src/test/resources/joined.vcf")

    val joinedPath = tmpDir.createTempFile("joined", "vds")

    val left = hc.importVCF("src/test/resources/joinleft.vcf")
    val right = hc.importVCF("src/test/resources/joinright.vcf")

    // make sure joined VDS writes
    left.unionCols(right).write(joinedPath)

    assert(joined.same(hc.readVDS(joinedPath)))
  }

  @Test def testIterator() {
    val leftVariants = Array(
      Variant("1", 1, "A", "T"),
      Variant("1", 2, "A", "T"),
      Variant("1", 4, "A", "T"),
      Variant("1", 4, "A", "T"),
      Variant("1", 5, "A", "T"),
      Variant("1", 5, "A", "T"),
      Variant("1", 9, "A", "T"),
      Variant("1", 13, "A", "T"),
      Variant("1", 13, "A", "T"),
      Variant("1", 14, "A", "T"),
      Variant("1", 15, "A", "T"))

    val rightVariants = Array(
      Variant("1", 1, "A", "T"),
      Variant("1", 1, "A", "T"),
      Variant("1", 1, "A", "T"),
      Variant("1", 3, "A", "T"),
      Variant("1", 4, "A", "T"),
      Variant("1", 4, "A", "T"),
      Variant("1", 6, "A", "T"),
      Variant("1", 6, "A", "T"),
      Variant("1", 8, "A", "T"),
      Variant("1", 9, "A", "T"),
      Variant("1", 13, "A", "T"),
      Variant("1", 15, "A", "T"))

    val vType = TVariant(GenomeReference.GRCh37)
    val leftKt = Table(hc, sc.parallelize(leftVariants.map(Row(_))), TStruct("v" -> vType)).keyBy("v")
    leftKt.typeCheck()
    val left = MatrixTable.fromRowsTable(leftKt)

    val rightKt = Table(hc, sc.parallelize(rightVariants.map(Row(_))), TStruct("v" -> vType)).keyBy("v")
    rightKt.typeCheck()
    val right = MatrixTable.fromRowsTable(rightKt)

    val localRowType = left.rvRowType

    // Inner distinct ordered join
    val jInner = left.rvd.orderedJoinDistinct(right.rvd, "inner")
    val jInnerOrdRDD1 = left.rdd.join(right.rdd.distinct)

    assert(jInner.count() == jInnerOrdRDD1.count())
    assert(jInner.forall(jrv => jrv.rvLeft != null && jrv.rvRight != null))
    assert(jInner.map { jrv =>
      val ur = new UnsafeRow(localRowType, jrv.rvLeft)
      ur.getAs[Variant](0)
    }.collect() sameElements jInnerOrdRDD1.map(_._1.asInstanceOf[Row].get(0)).collect().sorted(vType.ordering.toOrdering))

    // Left distinct ordered join
    val jLeft = left.rvd.orderedJoinDistinct(right.rvd, "left")
    val jLeftOrdRDD1 = left.rdd.leftOuterJoin(right.rdd.distinct)

    assert(jLeft.count() == jLeftOrdRDD1.count())
    assert(jLeft.forall(jrv => jrv.rvLeft != null))
    assert(jLeft.map { jrv =>
      val ur = new UnsafeRow(localRowType, jrv.rvLeft)
      ur.getAs[Variant](0)
    }.collect() sameElements jLeftOrdRDD1.map(_._1.asInstanceOf[Row].get(0)).collect().sorted(vType.ordering.toOrdering))
  }
}
