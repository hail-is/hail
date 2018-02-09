package is.hail.annotations

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.expr.types._
import is.hail.methods.SplitMulti
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.Variant
import org.apache.spark.sql.types._
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[is.hail.annotations]] package
  */
class AnnotationsSuite extends SparkSuite {
  @Test def test() {
    /*
      The below tests are designed to check for a subset of variants and info fields, that:
          1. the types, emitConversionIdentifier strings, and description strings agree with the VCF
          2. the strings stored in the AnnotationData classes agree with the VCF
          3. the strings stored in the AnnotationData classes convert correctly to the proper type
    */

    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val vas = vds.rowType
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - info.DP
    val dpQuery = vas.query("info", "DP")
    assert(vas.fieldOption("info", "DP").exists(_.typ.isInstanceOf[TInt32]))
    assert(dpQuery(variantAnnotationMap(firstVariant)) == 77560)
    assert(dpQuery(variantAnnotationMap(anotherVariant)) == 20271)

    // type Double - info.HWP
    val hwpQuery = vas.query("info", "HWP")
    assert(vas.fieldOption("info", "HWP").exists(_.typ.isInstanceOf[TFloat64]))
    assert(D_==(hwpQuery(variantAnnotationMap(firstVariant)).asInstanceOf[Double], 0.0001))
    assert(D_==(hwpQuery(variantAnnotationMap(anotherVariant)).asInstanceOf[Double], 0.8286))

    // type String - info.culprit
    val culpritQuery = vas.query("info", "culprit")
    assert(vas.fieldOption("info", "culprit").exists(_.typ.isInstanceOf[TString]))
    assert(culpritQuery(variantAnnotationMap(firstVariant)) == "FS")
    assert(culpritQuery(variantAnnotationMap(anotherVariant)) == "FS")

    // type Array - info.AC (allele count)
    val acQuery = vas.query("info", "AC")
    assert(vas.fieldOption("info", "AC").exists(f => f.typ == TArray(TInt32()) || f.typ == TArray(+TInt32())))
    assert(acQuery(variantAnnotationMap(firstVariant)) == IndexedSeq(89))
    assert(acQuery(variantAnnotationMap(anotherVariant)) == IndexedSeq(13))

    // type Boolean/flag - info.DB (dbSNP membership)
    val dbQuery = vas.query("info", "DB")
    assert(vas.fieldOption("info", "DB").exists(_.typ.isInstanceOf[TBoolean]))
    assert(dbQuery(variantAnnotationMap(firstVariant)) == true)
    assert(dbQuery(variantAnnotationMap(anotherVariant)) == null)

    //type Set[String]
    val filtQuery = vas.query("filters")
    assert(vas.fieldOption("filters").exists(f => f.typ == TSet(TString()) || f.typ == TSet(+TString())))
    assert(filtQuery(variantAnnotationMap(firstVariant)) == Set())
    assert(filtQuery(variantAnnotationMap(anotherVariant)) == Set("VQSRTrancheSNP99.95to100.00"))
  }

  @Test def testReadWrite() {
    val vds1 = hc.importVCF("src/test/resources/sample.vcf")
    val vds2 = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds1.same(vds2))

    val f = tmpDir.createTempFile("sample", extension = ".vds")
    vds1.write(f)
    val vds3 = hc.readVDS(f)
    assert(vds3.same(vds1))
  }


  @Test def testExtendedOrdering() {
    val ord = ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])
    val rord = ord.reverse

    assert(ord.lt(5, 7))
    assert(ord.lt(5, null))
    assert(ord.gt(null, 7))
    assert(ord.equiv(3, 3))
    assert(ord.equiv(null, null))
    assert(ord.max(5, 7) == 7)
    assert(ord.max(5, null) == null)
    assert(ord.min(5, 7) == 5)
    assert(ord.min(5, null) == 5)

    assert(rord.gt(5, 7))
    assert(rord.lt(5, null))
    assert(rord.gt(null, 7))
    assert(rord.max(5, 7) == 5)
    assert(rord.max(5, null) == null)
    assert(rord.min(5, 7) == 7)
    assert(rord.min(5, null) == 5)
  }
}
