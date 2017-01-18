package is.hail.io

import is.hail.utils._
import is.hail.driver._
import is.hail.io.vcf.LoadVCF
import is.hail.SparkSuite
import is.hail.utils.TempDir
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

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    ExportPlink.run(state, Array("-o", hailFile))

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

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/mendel.vcf"))
    s = SplitMulti.run(s)
    s = HardCalls.run(s)
    s = AnnotateSamplesFam.run(s, Array("-i", "src/test/resources/mendel.fam", "-d", "\\\\s+"))
    s = AnnotateSamplesExpr.run(s, Array("-c", "sa = sa.fam"))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va.rsid = str(v)"))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va = select(va, rsid)"))

    s = ExportPlink.run(s, Array("-o", plink, "-f",
      "famID = sa.famID, id = s.id, matID = sa.matID, patID = sa.patID, isFemale = sa.isFemale, isCase = sa.isCase"))

    var s2 = ImportPlink.run(s, Array("--bfile", plink))

    assert(s.vds.same(s2.vds))
  }
}
