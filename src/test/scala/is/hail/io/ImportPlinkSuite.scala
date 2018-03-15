package is.hail.io

import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.io.plink.{ExportPlink, LoadPlink}
import is.hail.methods.VariantQC
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import is.hail.{SparkSuite, TestUtils}
import org.testng.annotations.Test

import scala.language.postfixOps
import scala.sys.process._

class ImportPlinkSuite extends SparkSuite {

  object Spec extends Properties("ImportPlink") {
    val compGen = for {
      vds <- MatrixTable.gen(hc, VSMSubgen.random).map(vds => TestUtils.splitMultiHTS(vds).cache())
      nPartitions <- choose(1, LoadPlink.expectedBedSize(vds.numCols, vds.countRows()).toInt.min(10))
    } yield (vds, nPartitions)

    property("import generates same output as export") =
      forAll(compGen) { case (vds: MatrixTable, nPartitions: Int) =>

        val truthRoot = tmpDir.createTempFile("truth")
        val testRoot = tmpDir.createTempFile("test")

        ExportPlink(vds, truthRoot)

        if (vds.numCols == 0) {
          TestUtils.interceptFatal("Empty .fam file") {
            hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions), rg = Some(vds.referenceGenome))
          }
          true
        } else if (vds.countRows() == 0) {
          TestUtils.interceptFatal(".bim file does not contain any variants") {
            hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions))
          }
          true
        } else {
          ExportPlink(hc.importPlinkBFile(truthRoot, nPartitions = Some(nPartitions), rg = Some(vds.referenceGenome)),
            testRoot)

          val localTruthRoot = tmpDir.createLocalTempFile("truth")
          val localTestRoot = tmpDir.createLocalTempFile("test")

          hadoopConf.copy(truthRoot + ".fam", localTruthRoot + ".fam")
          hadoopConf.copy(truthRoot + ".bim", localTruthRoot + ".bim")
          hadoopConf.copy(truthRoot + ".bed", localTruthRoot + ".bed")

          hadoopConf.copy(testRoot + ".fam", localTestRoot + ".fam")
          hadoopConf.copy(testRoot + ".bim", localTestRoot + ".bim")
          hadoopConf.copy(testRoot + ".bed", localTestRoot + ".bed")

          val exitCodeFam = s"diff ${ uriPath(localTruthRoot) }.fam ${ uriPath(localTestRoot) }.fam" !
          val exitCodeBim = s"diff ${ uriPath(localTruthRoot) }.bim ${ uriPath(localTestRoot) }.bim" !
          val exitCodeBed = s"diff ${ uriPath(localTruthRoot) }.bed ${ uriPath(localTestRoot) }.bed" !

          exitCodeFam == 0 && exitCodeBim == 0 && exitCodeBed == 0
        }
      }

  }

  @Test def testPlinkImportRandom() {
    Spec.check()
  }

  @Test def testA1Major() {
    val plinkFileRoot = tmpDir.createTempFile("plink_reftest")
    ExportPlink(hc.importVCF("src/test/resources/sample.vcf"),
      plinkFileRoot)

    val a1ref = hc.importPlinkBFile(plinkFileRoot, a2Reference = false)

    val a2ref = hc.importPlinkBFile(plinkFileRoot, a2Reference = true)

    val a1kt = VariantQC(a1ref, "variant_qc")
      .rowsTable()
      .select(
        "row.rsid",
        "row.alleles",
        "row.variant_qc.n_not_called",
        "row.variant_qc.n_hom_ref",
        "row.variant_qc.n_het",
        "row.variant_qc.n_hom_var")
      .rename(Map("alleles" -> "vA1", "n_not_called" -> "nNotCalledA1",
        "n_hom_ref" -> "nHomRefA1", "n_het" -> "nHetA1", "n_hom_var" -> "nHomVarA1"), Map.empty[String, String])
      .keyBy("rsid")

    val a2kt = VariantQC(a2ref, "variant_qc")
      .rowsTable()
      .select(
        "row.rsid",
        "row.alleles",
        "row.variant_qc.n_not_called",
        "row.variant_qc.n_hom_ref",
        "row.variant_qc.n_het",
        "row.variant_qc.n_hom_var")
      .rename(Map("alleles" -> "vA2", "n_not_called" -> "nNotCalledA2",
        "n_hom_ref" -> "nHomRefA2", "n_het" -> "nHetA2", "n_hom_var" -> "nHomVarA2"), Map.empty[String, String])
      .keyBy("rsid")

    val joined = a1kt.join(a2kt, "outer")

    assert(joined.forall("row.vA1[0] == row.vA2[1] && " +
      "row.vA1[1] == row.vA2[0] && " +
      "row.nNotCalledA1 == row.nNotCalledA2 && " +
      "row.nHetA1 == row.nHetA2 && " +
      "row.nHomRefA1 == row.nHomVarA2 && " +
      "row.nHomVarA1 == row.nHomRefA2"))
  }
}
