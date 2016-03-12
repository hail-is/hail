package org.broadinstitute.hail.methods

import org.broadinstitute.hail.annotations.{SimpleSignature, Annotations}
import org.broadinstitute.hail.{expr, SparkSuite}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._

class FilterSuite extends SparkSuite {

  @Test def exprTest() {
    val symTab = Map("i" ->(0, expr.TInt),
      "j" ->(1, expr.TInt),
      "d" ->(2, expr.TDouble),
      "d2" ->(3, expr.TDouble),
      "s" ->(4, expr.TString),
      "s2" ->(5, expr.TString))
    val a = Array[Any](5, -7, 3.14, 5.79e7, "12,34,56,78",
      "this is a String, there are many like it, but this one is mine")

    def eval[T](s: String): T = {
      val f = expr.Parser.parse[T](symTab, a, s)
      f()
    }

    assert(eval[Int]("i") == 5)
    assert(eval[Int]("j") == -7)
    assert(eval[Int]("i.max(j)") == 5)
    assert(eval[Int]("i.min(j)") == -7)
    assert(D_==(eval[Double]("d"), 3.14))
    assert(eval[Array[String]]("""s.split(",")""") sameElements Array("12", "34", "56", "78"))
    assert(eval[Int]("s2.length") == 62)

    // FIXME catch parse errors
    // assert(eval[Boolean]("i.max(d) == 5"))
  }

  @Test def filterTest() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    var state = State(sc, sqlContext, vds.cache())
    state = SplitMulti.run(state, Array.empty[String])

    assert(FilterSamples.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nLocalSamples == 63)

    assert(FilterVariants.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    assert(FilterVariants.run(state, Array("--keep", "-c", "va.pass"))
      .vds.nVariants == 312)

    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AN == 200"))
      .vds.nVariants == 310)

    /*
    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AC.contains(20)"))
      .vds.nVariants == 3)
      */

    assert(FilterVariants.run(state, Array("--keep", "-c", """va.filters.contains("VQSRTrancheSNP99.60to99.80")"""))
      .vds.nVariants == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(FilterVariants.run(state, Array("--keep", "-c", """va.rsid != ".""""))
      .vds.nVariants == 258)

    assert(FilterVariants.run(state, Array("--remove", "-c", """va.rsid == ".""""))
      .vds.nVariants == 258)

    val stateWithSampleQC = SampleQC.run(state, Array.empty[String])

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.nCalled == 337"))
      .vds.nLocalSamples == 17)

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.dpMean > 60"))
      .vds.nLocalSamples == 7)

    assert(FilterSamples.run(stateWithSampleQC, Array("--keep", "-c", "if (\"^C1048\" ~ s.id) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3"))
      .vds.nLocalSamples == 14)

    val stateWithVariantQC = VariantQC.run(state, Array.empty[String])

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nCalled < 100"))
      .vds.nVariants == 36)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nHomVar > 0 && va.qc.nHet > 0"))
      .vds.nVariants == 104)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar > 0"))
      .vds.nVariants == 104)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar >= 0"))
      .vds.nVariants == 117)

    assert(FilterVariants.run(stateWithVariantQC, Array("--remove", "-c", "va.qc.rHetHomVar.isMissing"))
      .vds.nVariants == 117)

    assert(FilterVariants.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar.isNotMissing"))
      .vds.nVariants == 117)

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.gq.exists(_ < 20) })
    assert(highGQ.count { case (v, s, g) => g.gq.exists(_ >= 20) } == 30889)

    val highGQorMidQGAndLowFS = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20 || (g.gq < 30 && va.info.FS > 30)"))
      .vds.expand().collect()

    val vds2 = LoadVCF(sc, "src/test/resources/filter.vcf")
    var state2 = State(sc, sqlContext, vds2.cache())
    state2 = SplitMulti.run(state2, Array.empty[String])

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad(0) < 30")).vds.expand().collect().count(_._3.isCalled) == 3)

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad(1).toDouble / g.dp > 0.05")).vds.expand().collect().count(_._3.isCalled) == 3)

    val highGQ2 = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20"))

    assert(!highGQ2.vds.expand().collect().exists { case (v, s, g) => g.gq.exists(_ < 20) })

    val chr1 = FilterVariants.run(state2, Array("--keep", "-c", "v.contig == \"1\""))

    assert(chr1.vds.rdd.count == 9)

    assert(chr1.vds.expand().collect().count(_._3.isCalled) == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = FilterGenotypes.run(chr1, Array("--remove", "-c", "g.isHomRef"))
      .vds.expand().collect()

    assert(hetOrHomVarOnChr1.count(_._3.isCalled) == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = FilterGenotypes.run(chr1, Array("--keep", "-c", "g.isHomRef"))
      .vds.expand().collect()

    assert(homRefOnChr1.count(_._3.isCalled) == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes

  }

  @Test def filterFromFileTest() {

    val vds = TestRDDBuilder.buildRDD(8, 8, sc)

    var state = State(sc, sqlContext, vds)
    state = SplitMulti.run(state, Array.empty[String])

    assert(FilterSamples.run(state, Array("--keep", "-c", "src/test/resources/filter.sample_list")).vds.nLocalSamples == 3)

    assert(FilterSamples.run(state, Array("--remove", "-c", "src/test/resources/filter.sample_list")).vds.nLocalSamples == 5)

    assert(FilterVariants.run(state, Array("--keep", "-c", "src/test/resources/filter.interval_list")).vds.nVariants == 6)

    assert(FilterVariants.run(state, Array("--remove", "-c", "src/test/resources/filter.interval_list")).vds.nVariants == 2)

  }

  @Test def filterRegexTest() {
    val vds = LoadVCF(sc, "src/test/resources/multipleChromosomes.vcf")
    val s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val s2 = FilterVariants.run(s, Array("--keep", "-c", """ "^\\d+$" ~ v.contig """))
    assert(s.vds.nVariants == s2.vds.nVariants)
  }

  @Test def MissingTest() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val keepOneSample = FilterSamples.run(s, Array("--keep", "-c", "s.id == \"C1046::HG02024\""))
    val qc = VariantQC.run(keepOneSample, Array.empty[String])

    Count.run(keepOneSample, Array.empty[String])
    val missingVariants = qc.vds.variantsAndAnnotations
      .collect()
      .filter { case (v, va) =>
        va.get[Annotations]("qc").getOption[Double]("rHetHomVar")
          .isEmpty
      }
        .map(_._1)

    // ensure that we're not checking empty vs empty
    assert(missingVariants.size > 0)

    val missingVariantsFilter = FilterVariants.run(qc, Array("--keep", "-c", "va.qc.rHetHomVar.isMissing"))
      .vds
      .variantsAndAnnotations
      .collect()
      .map(_._1)

    assert(missingVariantsFilter.toSet == missingVariants.toSet)
  }

  @Test def testWeirdNames() {
    var vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    vds = vds
      .addVariantAnnotationSignatures("weird name \t test", SimpleSignature("Int"))
      .mapAnnotations((v, va) => va.+("weird name \t test", 1000))
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val s2 = FilterVariants.run(state, Array("--keep", "-c", "va.`weird name \t test` > 500"))
    assert(s2.vds.nVariants == vds.nVariants)
  }
}
