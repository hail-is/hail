package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations.UnsafeRow
import is.hail.expr.{TStruct, TVariant}
import is.hail.keytable.KeyTable
import is.hail.variant.{GenomeReference, Variant, VariantDataset}
import is.hail.utils._

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
    left.join(right).write(joinedPath)

    assert(joined.same(hc.readVDS(joinedPath)))
  }

  @Test def testIterator() {
    val leftVariants = Array(
      Variant("1", 1, "A", "T"),
      Variant("1", 2, "A", "T"),
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
      Variant("1", 3, "A", "T"),
      Variant("1", 4, "A", "T"),
      Variant("1", 6, "A", "T"),
      Variant("1", 6, "A", "T"),
      Variant("1", 8, "A", "T"),
      Variant("1", 9, "A", "T"),
      Variant("1", 13, "A", "T"),
      Variant("1", 15, "A", "T"))

    val leftKt = KeyTable(hc, sc.parallelize(leftVariants.map(Row(_))), TStruct("v" -> TVariant(GenomeReference.GRCh37))).keyBy("v")
    leftKt.typeCheck()
    val left = VariantDataset.fromKeyTable(leftKt)

    val rightKt = KeyTable(hc, sc.parallelize(rightVariants.map(Row(_))), TStruct("v" -> TVariant(GenomeReference.GRCh37))).keyBy("v")
    rightKt.typeCheck()
    val right = VariantDataset.fromKeyTable(rightKt)

    val localRowType = left.rowType

    // Inner distinct ordered join
    val jInner = left.rdd2.orderedJoinDistinct(right.rdd2, "inner")
    assert(jInner.forall(jrv => jrv.rvLeft != null && jrv.rvRight != null))
    assert(jInner.count() == 5)
    assert(jInner.map { jrv =>
      val ur = new UnsafeRow(localRowType, jrv.rvLeft)
      ur.getAs[Variant](1)
    }.collect() sameElements Array(Variant("1", 1, "A", "T"), Variant("1", 4, "A", "T"), Variant("1", 9, "A", "T"),
      Variant("1", 13, "A", "T"), Variant("1", 15, "A", "T")))

    // Left distinct ordered join
    val jLeft = left.rdd2.orderedJoinDistinct(right.rdd2, "left")
    assert(jLeft.forall(jrv => jrv.rvLeft != null))
    assert(jLeft.count() == left.variants.distinct().count())
    assert(jLeft.map { jrv =>
      val ur = new UnsafeRow(localRowType, jrv.rvLeft)
      ur.getAs[Variant](1)
    }.collect() sameElements left.variants.collect().distinct)
  }
}
