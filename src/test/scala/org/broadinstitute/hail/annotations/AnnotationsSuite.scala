package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.{Genotype, IntervalList, Variant}
import org.scalacheck.Gen
import org.testng.annotations.Test
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.methods.FilterUtils.toConvertibleString
import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[org.broadinstitute.hail.annotations]] package
  */
class AnnotationsSuite extends SparkSuite {
  @Test def test() {

    /*
      The below tests are designed to check for a subset of variants and info fields, that:
          1. the types, conversion strings, and description strings agree with the VCF
          2. the strings stored in the AnnotationData classes agree with the VCF
          3. the strings stored in the AnnotationData classes convert correctly to the proper type
    */

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)
    val vas = vds.metadata.variantAnnotationSignatures
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - INFO.DP
    assert(vas.get("info", "DP").contains(SimpleSignature("Int", "toInt",
      "Approximate read depth; some reads may have been filtered")))
    assert(variantAnnotationMap(firstVariant)
      .get("info", "DP")
      .contains("77560") &&
      variantAnnotationMap(firstVariant)
        .get("info", "DP").get.toInt == 77560)
    assert(variantAnnotationMap(anotherVariant)
      .get("info", "DP")
      .contains("20271") &&
      variantAnnotationMap(anotherVariant)
        .get("info", "DP").get.toInt == 20271)

    // type Double - INFO.HWP
    assert(vas.get("info", "HWP").contains(SimpleSignature("Double", "toDouble",
      "P value from test of Hardy Weinberg Equilibrium")))
    assert(variantAnnotationMap(firstVariant)
      .contains("info", "HWP") &&
      D_==(variantAnnotationMap(firstVariant)
        .get("info", "HWP").get.toDouble, 0.0001))
    assert(variantAnnotationMap(anotherVariant)
      .contains("info", "HWP") &&
      D_==(variantAnnotationMap(anotherVariant)
        .get("info", "HWP").get.toDouble, 0.8286))

    // type String - INFO.culprit
    assert(vas.get("info", "culprit").contains(SimpleSignature("String", "toString",
      "The annotation which was the worst performing in the Gaussian mixture model, " +
        "likely the reason why the variant was filtered out")))
    assert(variantAnnotationMap(firstVariant)
      .get("info", "culprit")
      .contains("FS"))
    assert(variantAnnotationMap(anotherVariant)
      .get("info", "culprit")
      .contains("FS"))

    // type Array - INFO.AC (allele count)
    assert(vas.get("info", "AC").contains(SimpleSignature("Array[Int]", "toArrayInt",
      "Allele count in genotypes, for each ALT allele, in the same order as listed")))
    assert(variantAnnotationMap(firstVariant)
      .get("info", "AC")
      .contains("89") &&
      variantAnnotationMap(firstVariant)
        .get("info", "AC").get.toArrayInt
        .sameElements(Array(89)))
    assert(variantAnnotationMap(anotherVariant)
      .get("info", "AC")
      .contains("13") &&
      variantAnnotationMap(anotherVariant)
        .get("info", "AC").get.toArrayInt
        .sameElements(Array(13)))

    // type Boolean/flag - INFO.DB (dbSNP membership)
    assert(vas.get("info", "DB").contains(SimpleSignature("Boolean", "toBoolean",
      "dbSNP Membership")))
    assert(variantAnnotationMap(firstVariant)
      .get("info", "DB")
      .contains("true") &&
      variantAnnotationMap(firstVariant)
        .get("info", "DB").get.toBoolean) // .get.toBoolean == true
    assert(!variantAnnotationMap(anotherVariant)
      .contains("info", "DB"))

    //type Set[String]
    assert(vas.get("filters").contains(SimpleSignature("Set[String]", "toSetString", "filters applied to site")))
    assert(variantAnnotationMap(firstVariant)
      .get("filters").contains("PASS") &&
      variantAnnotationMap(firstVariant)
      .get("filters").get.toSetString == Set[String]("PASS"))
    assert(variantAnnotationMap(anotherVariant)
      .get("filters").contains("VQSRTrancheSNP99.95to100.00") &&
      variantAnnotationMap(anotherVariant)
        .get("filters").get.toSetString == Set[String]("VQSRTrancheSNP99.95to100.00"))

    // GATK PASS
    assert(vas.get("pass").contains(SimpleSignature("Boolean", "toBoolean",
      "filters were applied to vcf and this site passed")))
    assert(variantAnnotationMap(firstVariant)
      .get("pass").contains("true"))
    assert(variantAnnotationMap(anotherVariant)
      .get("pass").contains("false"))
  }
}
