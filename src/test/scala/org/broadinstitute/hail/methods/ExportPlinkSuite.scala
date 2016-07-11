package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils._
import org.testng.annotations.Test

import scala.io.Source
import scala.language._
import scala.sys.process._

class ExportPlinkSuite extends SparkSuite {

  def rewriteBimIDs(file: String) {
    val parsed = readFile(file, sc.hadoopConfiguration) { is =>
      Source.fromInputStream(is)
        .getLines()
        .toList
        .map(line => line
          .split("\t"))
        .map(arr =>
          s"${arr(0)}\t${s"${arr(0)}:${arr(3)}:${arr(4)}:${arr(5)}"}\t${arr(2)}\t${arr(3)}\t${arr(4)}\t${arr(5)}\n")
    }
    writeTable(file, sc.hadoopConfiguration, parsed)
  }

  @Test def testBiallelic() {

    val localTmpDir = TempDir("file:///tmp", hadoopConf)

    val hailFile = localTmpDir.createTempFile("hail")
    val hailPath = uriPath(hailFile)

    val bFile = localTmpDir.createTempFile("plink")
    val bPath = uriPath(bFile)

    val mergeFile = localTmpDir.createTempFile("merge")
    val mergePath = uriPath(mergeFile)

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    vds.rdd.count
    ExportPlink.run(state, Array("-o", hailFile))

    // use plink to convert sample.vcf to a bed/bim/fam file
    s"plink --vcf src/test/resources/sample.vcf --make-bed --out $bPath --const-fid --keep-allele-order" !

    rewriteBimIDs(hailFile + ".bim")
    rewriteBimIDs(bFile + ".bim")

    // use plink to assert that the concordance rate is 1
    val exitCode = s"plink --bfile $bPath --bmerge $hailPath --merge-mode 6 --out $mergePath" !

    // assert that plink exited successfully
    assert(exitCode == 0)

    // assert that the .diff file is empty of non-header columns
    assert(
      readFile(mergeFile + ".diff", sc.hadoopConfiguration) { is =>
        Source.fromInputStream(is)
          .getLines()
          .toIndexedSeq
          .map(_.split(" +").filter(!_.isEmpty).toIndexedSeq) == IndexedSeq(IndexedSeq("SNP", "FID", "IID", "NEW", "OLD"))
      }
    )
  }
}
