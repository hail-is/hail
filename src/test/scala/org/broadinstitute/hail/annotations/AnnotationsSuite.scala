package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.{Genotype, IntervalList, Variant}
import org.testng.annotations.Test
import org.broadinstitute.hail.methods._
import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[org.broadinstitute.hail.annotations]] package
  */
class AnnotationsSuite extends SparkSuite {
  @Test def test() {

    /*
      The below tests are designed to check for a subset of variants and info fields, that:
          1. the types, emitConversionIdentifier strings, and description strings agree with the VCF
          2. the strings stored in the AnnotationData classes agree with the VCF
          3. the strings stored in the AnnotationData classes convert correctly to the proper type
    */

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")

    val state = State(sc, sqlContext, vds)
    val vas = vds.metadata.variantAnnotationSignatures
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - INFO.DP
    assert(vas.getAnnotations("info").attrs.get("DP").contains(VCFSignature("Int", "Integer", "1",
      "Approximate read depth; some reads may have been filtered")))
    assert(variantAnnotationMap(firstVariant)
        .getAnnotations("info").attrs.get("DP")
        .get.asInstanceOf[Int] == 77560)
    assert(variantAnnotationMap(anotherVariant)
      .getAnnotations("info").attrs.get("DP").get.asInstanceOf[Int] == 20271)

    // type Double - INFO.HWP
    assert(vas.getAnnotations("info").attrs.get("HWP").contains(new VCFSignature("Double", "Float", "1",
      "P value from test of Hardy Weinberg Equilibrium")))
    assert(
      D_==(variantAnnotationMap(firstVariant)
        .getAnnotations("info").attrs.get("HWP").get.asInstanceOf[Double], 0.0001))
    assert(D_==(variantAnnotationMap(anotherVariant)
        .getAnnotations("info").attrs.get("HWP").get.asInstanceOf[Double], 0.8286))

    // type String - INFO.culprit
    assert(vas.getAnnotations("info").attrs.get("culprit").contains(VCFSignature("String", "String", "1",
      "The annotation which was the worst performing in the Gaussian mixture model, " +
        "likely the reason why the variant was filtered out")))
    assert(variantAnnotationMap(firstVariant)
      .getAnnotations("info").attrs.get("culprit")
      .contains("FS"))
    assert(variantAnnotationMap(anotherVariant)
      .getAnnotations("info").attrs.get("culprit")
      .contains("FS"))

    // type Array - INFO.AC (allele count)
    assert(vas.getAnnotations("info").attrs.get("AC").contains(VCFSignature("IndexedSeq[Int]", "Integer", "A",
      "Allele count in genotypes, for each ALT allele, in the same order as listed")))
    assert(variantAnnotationMap(firstVariant)
      .getAnnotations("info").attrs.get("AC")
      .map(_.asInstanceOf[IndexedSeq[Int]])
      .forall(_.equals(IndexedSeq(89))))
    assert(variantAnnotationMap(anotherVariant)
      .getAnnotations("info").attrs.get("AC")
      .map(_.asInstanceOf[IndexedSeq[Int]])
      .forall(_.equals(IndexedSeq(13))))

    // type Boolean/flag - INFO.DB (dbSNP membership)
    assert(vas.getAnnotations("info").attrs.get("DB").contains(new VCFSignature("Boolean", "Flag", "0",
      "dbSNP Membership")))
    assert(variantAnnotationMap(firstVariant)
      .getAnnotations("info").attrs.get("DB")
      .contains(true))
    assert(!variantAnnotationMap(anotherVariant)
      .getAnnotations("info").attrs.contains("DB"))

    //type Set[String]
    assert(vas.attrs.get("filters").contains(new SimpleSignature("Set[String]")))
    assert(variantAnnotationMap(firstVariant)
      .attrs.get("filters").contains(Set[String]("PASS")))
    assert(variantAnnotationMap(anotherVariant)
      .attrs.get("filters").contains(Set("VQSRTrancheSNP99.95to100.00")))

    // GATK PASS
    assert(vas.attrs.get("pass").contains(new SimpleSignature("Boolean")))
    assert(variantAnnotationMap(firstVariant)
      .attrs.get("pass").contains(true))
    assert(variantAnnotationMap(anotherVariant)
      .attrs.get("pass").contains(false))

    val vds2 = LoadVCF(sc, "src/test/resources/sample2.vcf")


    // Check that VDS can be written to disk and retrieved while staying the same
    hadoopDelete("/tmp/sample.vds", sc.hadoopConfiguration, recursive = true)
    vds2.write(sqlContext, "/tmp/sample.vds")
    val readBack = Read.run(state, Array("-i", "/tmp/sample.vds"))

//    println(readBack.vds.metadata == vds2.metadata)
//    println(readBack.vds.variantsAndAnnotations.map{case (v, va) => va}.collect().toSet.diff(vds2.variantsAndAnnotations.map{case (v, va) => va}.collect.toSet))
//    println("read In = " + vds2.nVariants)
//    println("readBack = " + readBack.vds.nVariants)
    println(readBack.vds.eraseSplit.same(vds2.eraseSplit))
    println(readBack.vds.same(vds2))
    assert(readBack.vds.eraseSplit.same(vds2.eraseSplit))
  }
}
