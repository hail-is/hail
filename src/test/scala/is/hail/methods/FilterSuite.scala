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

    assert(vds.filterSamplesExpr("\"^HG\" ~ sa.s").numCols == 63)

    assert(vds.filterVariantsExpr("va.locus.position >= 14066228", keep = false).countVariants() == 173)

    assert(vds.filterVariantsExpr("va.filters.isEmpty").countVariants() == 312)

    assert(vds.filterVariantsExpr("va.info.AN == 200").countVariants() == 310)

    assert(vds.filterVariantsExpr("""va.filters.contains("VQSRTrancheSNP99.60to99.80")""").countVariants() == 3)

    // FIXME: rsid of "." should be treated as missing value
    assert(vds.filterVariantsExpr("""va.rsid != "."""").countVariants() == 258)

    assert(vds.filterVariantsExpr("""va.rsid == "."""", keep = false).countVariants() == 258)

    val sQcVds = SampleQC(vds)

    assert(sQcVds.filterSamplesExpr("sa.qc.nCalled == 337").numCols == 17)

    assert(sQcVds.filterSamplesExpr("sa.qc.dpMean > 60").numCols == 6)

    assert(sQcVds.filterSamplesExpr("if (\"^C1048\" ~ sa.s) {sa.qc.rTiTv > 3.5 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > 3")
      .numCols == 16)

    val vQcVds = VariantQC(vds)

    assert(vQcVds.filterVariantsExpr("va.qc.nCalled < 100").countVariants() == 36)

    assert(vQcVds.filterVariantsExpr("va.qc.nHomVar > 0 && va.qc.nHet > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar > 0").countVariants() == 104)

    assert(vQcVds.filterVariantsExpr("va.qc.rHetHomVar >= 0").countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isMissing(va.qc.rHetHomVar)", keep = false).countVariants() == 117)

    assert(vQcVds.filterVariantsExpr("isDefined(va.qc.rHetHomVar)").countVariants() == 117)

    val highGQ = vds.filterGenotypes("g.GQ < 20", keep = false)
    assert(!highGQ.entriesTable().exists("GQ < 20"))
    assert(highGQ.entriesTable().count() == 30889)

    val highGQorMidQGAndLowFS = vds.filterGenotypes("g.GQ < 20 || (g.GQ < 30 && va.info.FS > 30)", keep = false)
      .expand()
      .collect()

    val vds2 = SplitMulti(hc.importVCF("src/test/resources/filter.vcf"))
      .cache()

    assert(vds2.filterGenotypes("g.AD[0] < 30").entriesTable().count() == 3)

    assert(vds2.filterGenotypes("g.AD[1].toFloat64() / g.DP > 0.05")
        .entriesTable()
        .count() == 3)

    val highGQ2 = vds2.filterGenotypes("g.GQ < 20", keep = false)

    assert(!highGQ2.entriesTable().exists("GQ < 20"))

    val chr1 = vds2.filterVariantsExpr("va.locus.contig == \"1\"")

    assert(chr1.countVariants() == 9)
    assert(chr1.filterGenotypes("isDefined(g.GT)").entriesTable().count() == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = chr1.filterGenotypes("g.GT.isHomRef()", keep = false)

    assert(hetOrHomVarOnChr1.filterGenotypes("isDefined(g.GT)").entriesTable().count() == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = chr1.filterGenotypes("g.GT.isHomRef()")

    assert(homRefOnChr1.entriesTable().count() == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes
  }

  @Test def filterRegexTest() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/multipleChromosomes.vcf"))
    val vds2 = vds.filterVariantsExpr(""" "^\\d+$" ~ va.locus.contig """)
    assert(vds.countVariants() == vds2.countVariants())
  }

  @Test def MissingTest() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
    val keepOneSample = VariantQC(vds.filterSamplesExpr("sa.s == \"C1046::HG02024\""))

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
}
