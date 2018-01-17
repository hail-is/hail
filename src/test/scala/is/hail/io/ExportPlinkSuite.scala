package is.hail.io

import is.hail.SparkSuite
import is.hail.io.plink.FamFileConfig
import is.hail.keytable.KeyTable
import is.hail.utils._
import org.testng.annotations.Test

import scala.io.Source
import scala.language._
import scala.sys.process._

class ExportPlinkSuite extends SparkSuite {

  def rewriteBimIDs(file: String) {
    val parsed = hadoopConf.readFile(file) { is =>
      Source.fromInputStream(is)
        .getLines()
        .toList
        .map(line => line
          .split("\t"))
        .map(arr =>
          s"${ arr(0) }\t${ s"${ arr(0) }:${ arr(3) }:${ arr(4) }:${ arr(5) }" }\t${ arr(2) }\t${ arr(3) }\t${ arr(4) }\t${ arr(5) }\n")
    }
    hadoopConf.writeTable(file, parsed)
  }

  @Test def testBiallelic() {

    val hailFile = tmpDir.createTempFile("hail")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
    vds.exportPlink(hailFile)

    rewriteBimIDs(hailFile + ".bim")

    val localBFile = tmpDir.createLocalTempFile("plink")

    // use plink to convert sample.vcf to a bed/bim/fam file
    s"plink --vcf src/test/resources/sample.vcf --make-bed --out ${ uriPath(localBFile) } --const-fid --keep-allele-order" !

    rewriteBimIDs(localBFile + ".bim")

    val localMergeFile = tmpDir.createLocalTempFile("merge")

    val localHailFile = tmpDir.createLocalTempFile("hail")
    hadoopConf.copy(hailFile + ".bed", localHailFile + ".bed")
    hadoopConf.copy(hailFile + ".bim", localHailFile + ".bim")
    hadoopConf.copy(hailFile + ".fam", localHailFile + ".fam")

    // use plink to assert that the concordance rate is 1
    val exitCode = s"plink --bfile ${ uriPath(localBFile) } --bmerge ${ uriPath(localHailFile) } --merge-mode 6 --out ${ uriPath(localMergeFile) }" !

    // assert that plink exited successfully
    assert(exitCode == 0)

    // assert that the .diff file is empty of non-header columns
    assert(
      hadoopConf.readFile(localMergeFile + ".diff") { is =>
        Source.fromInputStream(is)
          .getLines()
          .toIndexedSeq
          .map(_.split(" +").filter(!_.isEmpty).toIndexedSeq) == IndexedSeq(IndexedSeq("SNP", "FID", "IID", "NEW", "OLD"))
      }
    )
  }

  @Test def testFamExport() {
    val plink = tmpDir.createTempFile("mendel")

    val vds = hc.importVCF("src/test/resources/mendel.vcf")
      .splitMulti()
      .hardCalls()
      .annotateSamplesTable(KeyTable.importFam(hc, "src/test/resources/mendel.fam", delimiter = "\\\\s+"), expr = "sa.fam = table")
      .annotateSamplesExpr("sa = sa.fam")
      .annotateVariantsExpr("va = {rsid: str(v)}")

    vds.exportPlink(plink,
      "famID = sa.famID, id = s, matID = sa.matID, patID = sa.patID, isFemale = sa.isFemale, isCase = sa.isCase")

    assert(hc.importPlinkBFile(plink).same(vds))
  }

  @Test def testParallelExport() {
    val fParallel = tmpDir.createTempFile("exportPlinkParallel")

    val vds = hc.importVCF("src/test/resources/sample.vcf", nPartitions = Some(2)).splitMulti()
    vds.exportPlink(fParallel, parallel = true)
    hc.importPlink(fParallel + ".bed/part-00000", fParallel + ".bim/part-00000", fParallel + ".fam").count()
    hc.importPlink(fParallel + ".bed/part-00001", fParallel + ".bim/part-00001", fParallel + ".fam").count()

    val localHailFile = tmpDir.createLocalTempFile("hail")

    val Array(bed0, bim0, fam0) = Array(".bed/part-00000", ".bim/part-00000", ".fam").map { s =>
      hadoopConf.copy(fParallel + s, localHailFile + s)
      uriPath(localHailFile + s)
    }

    val Array(bed1, bim1, fam1) = Array(".bed/part-00001", ".bim/part-00001", ".fam").map { s =>
      hadoopConf.copy(fParallel + s, localHailFile + s)
      uriPath(localHailFile + s)
    }

    val localMergeBFile = tmpDir.createLocalTempFile("plink")
    s"plink --bed $bed0 --bim $bim0 --fam $fam0 --bmerge $bed1 $bim1 $fam1 --make-bed --indiv-sort 0 --const-fid --keep-allele-order --out ${ uriPath(localMergeBFile) }" !

    val notParallel = tmpDir.createTempFile("exportPlinkNotParallel")
    vds.exportPlink(notParallel, parallel = false)

    assert(hc.importPlinkBFile(notParallel).same(hc.importPlinkBFile(localMergeBFile)))
  }
}
