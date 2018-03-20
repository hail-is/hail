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
      .select(Array(
        "row.rsid",
        "alleles_a1 = row.alleles",
        "n_not_called_a1 = row.variant_qc.n_not_called",
        "n_hom_ref_a1 = row.variant_qc.n_hom_ref",
        "n_het_a1 = row.variant_qc.n_het",
        "n_hom_var_a1 = row.variant_qc.n_hom_var"))
      .keyBy("rsid")

    val a2kt = VariantQC(a2ref, "variant_qc")
      .rowsTable()
      .select(Array(
        "row.rsid",
        "alleles_a2 = row.alleles",
        "n_not_called_a2 = row.variant_qc.n_not_called",
        "n_hom_ref_a2 = row.variant_qc.n_hom_ref",
        "n_het_a2 = row.variant_qc.n_het",
        "n_hom_var_a2 = row.variant_qc.n_hom_var"))
      .keyBy("rsid")

    val joined = a1kt.join(a2kt, "outer")

    assert(joined.forall(
      "row.alleles_a1[0] == row.alleles_a2[1] && " +
        "row.alleles_a1[1] == row.alleles_a2[0] && " +
        "row.n_not_called_a1 == row.n_not_called_a2 && " +
        "row.n_het_a1 == row.n_het_a2 && " +
        "row.n_hom_ref_a1 == row.n_hom_var_a2 && " +
        "row.n_hom_var_a1 == row.n_hom_ref_a2"))
  }
}
