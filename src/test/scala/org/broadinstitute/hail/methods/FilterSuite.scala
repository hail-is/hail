package org.broadinstitute.hail.methods

import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.broadinstitute.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {

  @Test def filterTest() {

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    var state = State(sc, sqlContext, vds.cache())
    state = SplitMulti.run(state, Array.empty[String])

    assert(FilterSamplesExpr.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nSamples == 63)

    assert(FilterVariantsExpr.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    assert(FilterVariantsExpr.run(state, Array("--keep", "-c", "va.pass"))
      .vds.nVariants == 312)

    assert(FilterVariantsExpr.run(state, Array("--keep", "-c", "va.info.AN == 200"))
      .vds.nVariants == 310)

    /*
    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AC.contains(20)"))
      .vds.nVariants == 3)
      */

    assert(FilterVariantsExpr.run(state, Array("--keep", "-c", """va.filters.contains("VQSRTrancheSNP99.60to99.80")"""))
      .vds.nVariants == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(FilterVariantsExpr.run(state, Array("--keep", "-c", """va.rsid != ".""""))
      .vds.nVariants == 258)

    assert(FilterVariantsExpr.run(state, Array("--remove", "-c", """va.rsid == ".""""))
      .vds.nVariants == 258)

    val stateWithSampleQC = SampleQC.run(state, Array.empty[String])

    assert(FilterSamplesExpr.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.nCalled == 337"))
      .vds.nSamples == 17)

    assert(FilterSamplesExpr.run(stateWithSampleQC, Array("--keep", "-c", "sa.qc.dpMean > 60"))
      .vds.nSamples == 7)

    assert(FilterSamplesExpr.run(stateWithSampleQC, Array("--keep", "-c", "if (\"^C1048\" ~ s.id) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3"))
      .vds.nSamples == 14)

    val stateWithVariantQC = VariantQC.run(state, Array.empty[String])

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nCalled < 100"))
      .vds.nVariants == 36)

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.nHomVar > 0 && va.qc.nHet > 0"))
      .vds.nVariants == 104)

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar > 0"))
      .vds.nVariants == 104)

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--keep", "-c", "va.qc.rHetHomVar >= 0"))
      .vds.nVariants == 117)

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--remove", "-c", "isMissing(va.qc.rHetHomVar)"))
      .vds.nVariants == 117)

    assert(FilterVariantsExpr.run(stateWithVariantQC, Array("--keep", "-c", "isDefined(va.qc.rHetHomVar)"))
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

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad[0] < 30")).vds.expand().collect().count(_._3.isCalled) == 3)

    assert(FilterGenotypes.run(state2, Array("--keep", "-c", "g.ad[1].toDouble / g.dp > 0.05")).vds.expand().collect().count(_._3.isCalled) == 3)

    val highGQ2 = FilterGenotypes.run(state, Array("--remove", "-c", "g.gq < 20"))

    assert(!highGQ2.vds.expand().collect().exists { case (v, s, g) => g.gq.exists(_ < 20) })

    val chr1 = FilterVariantsExpr.run(state2, Array("--keep", "-c", "v.contig == \"1\""))

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

    assert(FilterSamplesList.run(state, Array("--keep", "-i", "src/test/resources/filter.sample_list")).vds.nSamples == 3)

    assert(FilterSamplesList.run(state, Array("--remove", "-i", "src/test/resources/filter.sample_list")).vds.nSamples == 5)

    assert(FilterVariantsIntervals.run(state, Array("--keep", "-i", "src/test/resources/filter.interval_list")).vds.nVariants == 6)

    assert(FilterVariantsIntervals.run(state, Array("--remove", "-i", "src/test/resources/filter.interval_list")).vds.nVariants == 2)

  }

  @Test def filterRegexTest() {
    val vds = LoadVCF(sc, "src/test/resources/multipleChromosomes.vcf")
    val s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val s2 = FilterVariantsExpr.run(s, Array("--keep", "-c", """ "^\\d+$" ~ v.contig """))
    assert(s.vds.nVariants == s2.vds.nVariants)
  }

  @Test def MissingTest() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val keepOneSample = FilterSamplesExpr.run(s, Array("--keep", "-c", "s.id == \"C1046::HG02024\""))
    val qc = VariantQC.run(keepOneSample, Array.empty[String])

    val q = qc.vds.queryVA("va.qc.rHetHomVar")._2
    val missingVariants = qc.vds.variantsAndAnnotations
      .collect()
      .filter { case (v, va) =>
        q(va).isEmpty
      }
      .map(_._1)

    // ensure that we're not checking empty vs empty
    assert(missingVariants.size > 0)

    val missingVariantsFilter = FilterVariantsExpr.run(qc, Array("--keep", "-c", "isMissing(va.qc.rHetHomVar)"))
      .vds
      .variantsAndAnnotations
      .collect()
      .map(_._1)

    assert(missingVariantsFilter.toSet == missingVariants.toSet)
  }

  @Test def testWeirdNames() {
    var vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val (sigs, i) = vds.insertVA(TInt, "weird name \t test")
    vds = vds
      .mapAnnotations((v, va, gs) => i(va, Some(1000)))
      .copy(vaSignature = sigs)
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val s2 = FilterVariantsExpr.run(state, Array("--keep", "-c", "va.`weird name \\t test` > 500"))
    assert(s2.vds.nVariants == vds.nVariants)

    TestUtils.interceptFatal("invalid escape character.*backtick identifier.*\\\\i")(
      FilterVariantsExpr.run(state, Array("--keep", "-c", "va.`bad\\input` == 5")))
  }

  @Test def testPAB() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val s2 = FilterGenotypes.run(state, Array("--keep", "-c", "g.isHet && g.pAB > 0.0005"))
    s2.vds.expand()
      .collect()
      .foreach { case (v, s, g) =>
        assert(!g.isHet || g.pAB().forall(_ > 0.0005))
      }
  }
}
