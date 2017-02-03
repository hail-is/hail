package is.hail.methods

import is.hail.driver._
import is.hail.expr._
import is.hail.io.vcf.LoadVCF
import is.hail.utils.TestRDDBuilder
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {

  @Test def filterTest() {

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()

    assert(vds.filterSamplesExpr("\"^HG\" ~ s.id").nSamples == 63)

    assert(vds.filterVariantsExpr("v.start >= 14066228", remove = true).countVariants() == 173)

    assert(vds.filterVariantsExpr("va.pass").countVariants() == 312)

    assert(vds.filterVariantsExpr("va.info.AN == 200").countVariants() == 310)

    /*
    assert(FilterVariants.run(state, Array("--keep", "-c", "va.info.AC.contains(20)"))
      .vds.nVariants == 3)
      */

    assert(vds.filterVariantsExpr("""va.filters.contains("VQSRTrancheSNP99.60to99.80")""").countVariants() == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(vds.filterVariantsExpr("""va.rsid != "."""").countVariants() == 258)

    assert(vds.filterVariantsExpr("""va.rsid == "."""", remove = true).countVariants() == 258)

    val sQcVds = vds.sampleQC()

    assert(sQcVds.filterSamplesExpr("sa.qc.nCalled == 337").nSamples == 17)

    assert(sQcVds.filterSamplesExpr("sa.qc.dpMean > 60").nSamples == 7)

    assert(sQcVds.filterSamplesExpr("if (\"^C1048\" ~ s.id) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3")
      .nSamples == 16)

    val vQcVds = vds.variantQC()

    assert(vQcVds.filterVariantsExpr("va.qc.nCalled < 100").countVariants() == 36)

    assert(vQcVds.filterVariantsExpr("va.qc.nHomVar > 0 && va.qc.nHet > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar >= 0").countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isMissing(va.qc.rHetHomVar)", remove = true).countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isDefined(va.qc.rHetHomVar)").countVariants() == 117)

    val highGQ = vds.filterGenotypes("g.gq < 20", remove = true)
      .expand()
      .collect()

    assert(!highGQ.exists { case (v, s, g) => g.gq.exists(_ < 20) })
    assert(highGQ.count { case (v, s, g) => g.gq.exists(_ >= 20) } == 30889)

    val highGQorMidQGAndLowFS = vds.filterGenotypes("g.gq < 20 || (g.gq < 30 && va.info.FS > 30)", remove = true)
      .expand()
      .collect()

    val vds2 = hc.importVCF("src/test/resources/filter.vcf")
      .cache()
      .splitMulti()

    assert(vds2.filterGenotypes("g.ad[0] < 30").expand().collect().count(_._3.isCalled) == 3)

    assert(vds2.filterGenotypes("g.ad[1].toDouble / g.dp > 0.05").expand().collect().count(_._3.isCalled) == 3)

    val highGQ2 = vds2.filterGenotypes("g.gq < 20", remove = true)

    assert(!highGQ2.expand().collect().exists { case (v, s, g) => g.gq.exists(_ < 20) })

    val chr1 = vds2.filterVariantsExpr("v.contig == \"1\"")

    assert(chr1.rdd.count == 9)

    assert(chr1.expand().collect().count(_._3.isCalled) == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = chr1.filterGenotypes("g.isHomRef", remove = true)
      .expand()
      .collect()

    assert(hetOrHomVarOnChr1.count(_._3.isCalled) == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = chr1.filterGenotypes("g.isHomRef")
      .expand()
      .collect()

    assert(homRefOnChr1.count(_._3.isCalled) == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes
  }

  @Test def filterFromFileTest() {

    val vds = TestRDDBuilder.buildRDD(8, 8, hc)
      .splitMulti()

    assert(vds.filterSamplesList("src/test/resources/filter.sample_list").nSamples == 3)

    assert(vds.filterSamplesList("src/test/resources/filter.sample_list", remove = true).nSamples == 5)

    assert(vds.filterIntervals("src/test/resources/filter.interval_list", keep = true).countVariants() == 6)

    assert(vds.filterIntervals("src/test/resources/filter.interval_list", keep = false).countVariants() == 2)
  }

  @Test def filterRegexTest() {
    val vds = hc.importVCF("src/test/resources/multipleChromosomes.vcf")
      .splitMulti()
    val vds2 = vds.filterVariantsExpr(""" "^\\d+$" ~ v.contig """)
    assert(vds.countVariants() == vds2.countVariants())
  }

  @Test def MissingTest() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
    val keepOneSample = vds.filterSamplesExpr("s.id == \"C1046::HG02024\"")
      .variantQC()

    val q = keepOneSample.queryVA("va.qc.rHetHomVar")._2
    val missingVariants = keepOneSample.variantsAndAnnotations
      .collect()
      .filter { case (v, va) =>
        q(va).isEmpty
      }
      .map(_._1)

    // ensure that we're not checking empty vs empty
    assert(missingVariants.size > 0)

    val missingVariantsFilter = keepOneSample.filterVariantsExpr("isMissing(va.qc.rHetHomVar)")
      .variantsAndAnnotations
      .collect()
      .map(_._1)

    assert(missingVariantsFilter.toSet == missingVariants.toSet)
  }

  @Test def testWeirdNames() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")
    val (sigs, i) = vds.insertVA(TInt, "weird name \t test")
    vds = vds
      .mapAnnotations((v, va, gs) => i(va, Some(1000)))
      .copy(vaSignature = sigs)
    assert(vds.filterVariantsExpr("va.`weird name \\t test` > 500").countVariants() == vds.countVariants)

    TestUtils.interceptFatal("invalid escape character.*backtick identifier.*\\\\i")(
      vds.filterVariantsExpr("va.`bad\\input` == 5"))
  }

  @Test def testPAB() {
    hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
      .filterGenotypes("g.isHet && g.pAB > 0.0005")
      .expand()
      .collect()
      .foreach { case (v, s, g) =>
        assert(!g.isHet || g.pAB().forall(_ > 0.0005))
      }
  }
}
