package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
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

    def vasFieldSig(field: String): Option[Type] =
      vas.castOption[TStruct]
        .flatMap(_.get(field))

    def infoFieldSig(field: String): Option[Type] =
      vasFieldSig("info")
        .flatMap(_.castOption[TStruct])
        .flatMap(_.get(field))

    // type Int - INFO.DP
    println(vas("info.DP").o)
    assert(vas("info.DP").contains(TInt))
    assert(variantAnnotationMap(firstVariant)
      .get[Annotations]("info").attrs.get("DP")
      .get.asInstanceOf[Int] == 77560)
    assert(variantAnnotationMap(anotherVariant)
      .get[Annotations]("info").attrs.get("DP").get.asInstanceOf[Int] == 20271)

    // type Double - INFO.HWP
    assert(infoFieldSig("HWP").contains(TDouble))
    assert(
      D_==(variantAnnotationMap(firstVariant)
        .get[Annotations]("info").attrs.get("HWP").get.asInstanceOf[Double], 0.0001))
    assert(D_==(variantAnnotationMap(anotherVariant)
      .get[Annotations]("info").attrs.get("HWP").get.asInstanceOf[Double], 0.8286))

    // type String - INFO.culprit
    assert(infoFieldSig("culprit").contains(TString))
    assert(variantAnnotationMap(firstVariant)
      .get[Annotations]("info").attrs.get("culprit")
      .contains("FS"))
    assert(variantAnnotationMap(anotherVariant)
      .get[Annotations]("info").attrs.get("culprit")
      .contains("FS"))

    // type Array - INFO.AC (allele count)
    assert(infoFieldSig("AC").contains(TArray(TInt)))
    assert(variantAnnotationMap(firstVariant)
      .get[Annotations]("info").attrs.get("AC")
      .map(_.asInstanceOf[IndexedSeq[Int]])
      .forall(_.sameElements(IndexedSeq(89))))
    assert(variantAnnotationMap(anotherVariant)
      .get[Annotations]("info").attrs.get("AC")
      .map(_.asInstanceOf[IndexedSeq[Int]])
      .forall(_.equals(IndexedSeq(13))))

    // type Boolean/flag - INFO.DB (dbSNP membership)
    assert(infoFieldSig("DB").contains(TBoolean))
    assert(variantAnnotationMap(firstVariant)
      .get[Annotations]("info").attrs.get("DB")
      .contains(true))
    assert(!variantAnnotationMap(anotherVariant)
      .get[Annotations]("info").attrs.contains("DB"))

    //type Set[String]
    assert(vasFieldSig("filters").contains(TSet(TString)))
    assert(variantAnnotationMap(firstVariant)
      .attrs.get("filters").contains(Set[String]("PASS")))
    assert(variantAnnotationMap(anotherVariant)
      .attrs.get("filters").contains(Set("VQSRTrancheSNP99.95to100.00")))

    // GATK PASS
    assert(vasFieldSig("pass").contains(TBoolean))
    assert(variantAnnotationMap(firstVariant)
      .attrs.get("pass").contains(true))
    assert(variantAnnotationMap(anotherVariant)
      .attrs.get("pass").contains(false))

    val vds2 = LoadVCF(sc, "src/test/resources/sample2.vcf")


    // Check that VDS can be written to disk and retrieved while staying the same
    hadoopDelete("/tmp/sample.vds", sc.hadoopConfiguration, recursive = true)
    vds2.write(sqlContext, "/tmp/sample.vds")
    val readBack = Read.run(state, Array("-i", "/tmp/sample.vds"))

    assert(readBack.vds.same(vds2))
  }

  @Test def testMergeAnnotations() {
    val map1 = Map[String, Any]("a" -> 1, "b" -> 2, "c" -> 3)
    val map2 = Map[String, Any]("a" -> 4, "b" -> 5)
    val map3 = Map[String, Any]("a" -> 6)

    val anno1 = Annotations(Map("a" -> 1))
    val anno2 = Annotations(Map("a" -> Annotations(map2), "b" -> 2, "c" -> 3))
    val anno3 = Annotations(Map("a" -> Annotations(Map("a" -> 1))))

    // make sure that adding one deep annotation does the right thing
    // make sure that overwriting one high annotation does the right thing
    assert(anno2 ++ anno1 == Annotations(map1))
    assert(anno2 ++ anno3 == Annotations(Map("a" -> Annotations(Map("a" -> 1, "b" -> 5)), "b" -> 2, "c" -> 3)))
  }
}
