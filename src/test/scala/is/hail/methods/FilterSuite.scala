package is.hail.methods

import is.hail.testUtils._
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {
  @Test def filterTest() {
    val vds = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/sample.vcf"))

    assert(vds.filterColsExpr("\"^HG\" ~ sa.s").numCols == 63)

    assert(vds.filterRowsExpr("va.locus.position >= 14066228", keep = false).countRows() == 173)

    assert(vds.filterRowsExpr("va.filters.isEmpty").countRows() == 312)

    assert(vds.filterRowsExpr("va.info.AN == 200").countRows() == 310)

    assert(vds.filterRowsExpr("""va.filters.contains("VQSRTrancheSNP99.60to99.80")""").countRows() == 3)

    assert(vds.filterRowsExpr("""isDefined(va.rsid)""").countRows() == 258)

    assert(vds.filterRowsExpr("""isMissing(va.rsid)""", keep = false).countRows() == 258)

    val sQcVds = SampleQC(vds)

    assert(sQcVds.filterColsExpr("sa.qc.n_called == 337").numCols == 17)

    assert(sQcVds.filterColsExpr("sa.qc.dp_mean > 60d").numCols == 6)

    assert(sQcVds.filterColsExpr("if (\"^C1048\" ~ sa.s) {sa.qc.r_ti_tv > 3.5 && sa.qc.n_singleton < 10000000L} else sa.qc.r_ti_tv > 3d")
      .numCols == 16)

    val vQcVds = VariantQC(vds)

    assert(vQcVds.filterRowsExpr("va.qc.n_called < 100").countRows() == 36)

    assert(vQcVds.filterRowsExpr("va.qc.n_hom_var > 0 && va.qc.n_het > 0").countRows() == 104)

    assert(vQcVds.filterRowsExpr("va.qc.r_het_hom_var > 0d").countRows() == 104)

    assert(vQcVds.filterRowsExpr("va.qc.r_het_hom_var >= 0d").countRows() == 117)

    assert(vQcVds.filterRowsExpr("isMissing(va.qc.r_het_hom_var)", keep = false).countRows() == 117)

    assert(vQcVds.filterRowsExpr("isDefined(va.qc.r_het_hom_var)").countRows() == 117)

    val highGQ = vds.filterEntries("g.GQ < 20", keep = false)
    assert(!highGQ.entriesTable().exists("row.GQ < 20"))
    assert(highGQ.entriesTable().count() == 30889)

    val highGQorMidQGAndLowFS = vds.filterEntries("g.GQ < 20 || (g.GQ < 30 && va.info.FS > 30d)", keep = false)
      .expand()
      .collect()

    val vds2 = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/filter.vcf"))
      .cache()

    assert(vds2.filterEntries("g.AD[0] < 30").entriesTable().count() == 3)

    assert(vds2.filterEntries("g.AD[1] / g.DP > 0.05f")
        .entriesTable()
        .count() == 3)

    val highGQ2 = vds2.filterEntries("g.GQ < 20", keep = false)

    assert(!highGQ2.entriesTable().exists("row.GQ < 20"))

    val chr1 = vds2.filterRowsExpr("va.locus.contig == \"1\"")

    assert(chr1.countRows() == 9)
    assert(chr1.filterEntries("isDefined(g.GT)").entriesTable().count() == 9 * 11 - 2)

    val hetOrHomVarOnChr1 = chr1.filterEntries("g.GT.isHomRef()", keep = false)

    assert(hetOrHomVarOnChr1.filterEntries("isDefined(g.GT)").entriesTable().count() == 9 + 3 + 3) // remove does not retain the 2 missing genotypes

    val homRefOnChr1 = chr1.filterEntries("g.GT.isHomRef()")

    assert(homRefOnChr1.entriesTable().count() == 9 * 11 - (9 + 3 + 3) - 2) // keep does not retain the 2 missing genotypes
  }

  @Test def filterRegexTest() {
    val vds = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/multipleChromosomes.vcf"))
    val vds2 = vds.filterRowsExpr(""" "^\\d+$" ~ va.locus.contig """)
    assert(vds.countRows() == vds2.countRows())
  }

  @Test def MissingTest() {
    val vds = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/sample.vcf"))
    val keepOneSample = VariantQC(vds.filterColsExpr("sa.s == \"C1046::HG02024\""))

    val q = keepOneSample.queryVA("va.qc.r_het_hom_var")._2
    val missingVariants = keepOneSample.variantsAndAnnotations
      .collect()
      .filter { case (v, va) =>
        q(va) == null
      }
      .map(_._1)

    // ensure that we're not checking empty vs empty
    assert(missingVariants.size > 0)

    val missingVariantsFilter = keepOneSample.filterRowsExpr("isMissing(va.qc.r_het_hom_var)")
      .variantsAndAnnotations
      .collect()
      .map(_._1)

    assert(missingVariantsFilter.toSet == missingVariants.toSet)
  }
}
