package is.hail.methods

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.expr.types._
import is.hail.io.annotators.IntervalList
import is.hail.utils._
import is.hail.testUtils._
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {
  @Test def filterTest() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))

    assert(vds.filterSamplesExpr("\"^HG\" ~ s").nSamples == 63)

    assert(vds.filterVariantsExpr("v.start >= 14066228", keep = false).countVariants() == 173)

    assert(vds.filterVariantsExpr("va.filters.isEmpty").countVariants() == 312)

    assert(vds.filterVariantsExpr("va.info.AN == 200").countVariants() == 310)

    TestUtils.interceptFatal("out-of-place expression") {
      vds.filterVariantsExpr("x => v.altAllele.isSNP()")
        .countVariants() // force evaluation
    }

    assert(vds.filterVariantsExpr("""va.filters.contains("VQSRTrancheSNP99.60to99.80")""").countVariants() == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(vds.filterVariantsExpr("""va.rsid != "."""").countVariants() == 258)

    assert(vds.filterVariantsExpr("""va.rsid == "."""", keep = false).countVariants() == 258)

    val sQcVds = SampleQC(vds)

    assert(sQcVds.filterSamplesExpr("sa.qc.nCalled == 337").nSamples == 17)

    assert(sQcVds.filterSamplesExpr("sa.qc.dpMean > 60").nSamples == 6)

    assert(sQcVds.filterSamplesExpr("if (\"^C1048\" ~ s) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3")
      .nSamples == 16)

    val vQcVds = VariantQC(vds)

    assert(vQcVds.filterVariantsExpr("va.qc.nCalled < 100").countVariants() == 36)

    assert(vQcVds.filterVariantsExpr("va.qc.nHomVar > 0 && va.qc.nHet > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar >= 0").countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isMissing(va.qc.rHetHomVar)", keep = false).countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isDefined(va.qc.rHetHomVar)").countVariants() == 117)

    val highGQ = vds.filterGenotypes("g.GQ < 20", keep = false)
    assert(!highGQ.genotypeKT().exists("g.GQ < 20"))
    assert(highGQ.genotypeKT().count() == 30889)

    val highGQorMidQGAndLowFS = vds.filterGenotypes("g.GQ < 20 || (g.GQ < 30 && va.info.FS > 30)", keep = false)
      .expand()
      .collect()

    val vds2 = SplitMulti(hc.importVCF("src/test/resources/filter.vcf"))
      .cache()

    assert(vds2.filterGenotypes("g.AD[0] < 30").genotypeKT().count() == 3)

    assert(vds2.filterGenotypes("g.AD[1].toFloat64() / g.DP > 0.05")
        .genotypeKT()
        .count() == 3)

    val highGQ2 = vds2.filterGenotypes("g.GQ < 20", keep = false)

    assert(!highGQ2.genotypeKT().exists("g.GQ < 20"))

    val chr1 = vds2.filterVariantsExpr("v.contig == \"1\"")

    assert(chr1.countVariants() == 9)
    assert(chr1.filterGenotypes("isDefined(g.GT)").genotypeKT().count() == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = chr1.filterGenotypes("g.GT.isHomRef()", keep = false)

    assert(hetOrHomVarOnChr1.filterGenotypes("isDefined(g.GT)").genotypeKT().count() == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = chr1.filterGenotypes("g.GT.isHomRef()")

    assert(homRefOnChr1.genotypeKT().count() == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes
  }

  @Test def filterFromFileTest() {

    val vds = SplitMulti(TestRDDBuilder.buildRDD(8, 8, hc))

    val sampleList = hadoopConf.readLines("src/test/resources/filter.sample_list")(_.map(_.value: Annotation).toSet)
    assert(vds.filterSamplesList(sampleList).nSamples == 3)

    assert(vds.filterSamplesList(sampleList, keep = false).nSamples == 5)

    assert(vds.filterVariantsTable(IntervalList.read(hc, "src/test/resources/filter.interval_list"), keep = true)
      .countVariants() == 6)

    assert(vds.filterVariantsTable(IntervalList.read(hc, "src/test/resources/filter.interval_list"), keep = false)
      .countVariants() == 2)
  }

  @Test def filterRegexTest() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/multipleChromosomes.vcf"))
    val vds2 = vds.filterVariantsExpr(""" "^\\d+$" ~ v.contig """)
    assert(vds.countVariants() == vds2.countVariants())
  }

  @Test def MissingTest() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
    val keepOneSample = VariantQC(vds.filterSamplesExpr("s == \"C1046::HG02024\""))

    val q = keepOneSample.queryVA("va.qc.rHetHomVar")._2
    val missingVariants = keepOneSample.variantsAndAnnotations
      .collect()
      .filter { case (v, va) =>
        q(va) == null
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
    val (sigs, i) = vds.insertVA(TInt32(), "weird name \t test")
    vds = vds
      .mapAnnotations(sigs, (v, va, gs) => i(va, 1000))
    assert(vds.filterVariantsExpr("va.`weird name \\t test` > 500").countVariants() == vds.countVariants)

    TestUtils.interceptFatal("invalid escape character.*backtick identifier.*\\\\i")(
      vds.filterVariantsExpr("va.`bad\\input` == 5"))
  }
}
