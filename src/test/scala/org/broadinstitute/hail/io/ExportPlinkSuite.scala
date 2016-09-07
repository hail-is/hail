package org.broadinstitute.hail.io

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.vcf.LoadVCF
import org.broadinstitute.hail.{SparkSuite, TempDir}
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
          s"${ arr(0) }\t${ s"${ arr(0) }:${ arr(3) }:${ arr(4) }:${ arr(5) }" }\t${ arr(2) }\t${ arr(3) }\t${ arr(4) }\t${ arr(5) }\n")
    }
    writeTable(file, sc.hadoopConfiguration, parsed)
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
    hadoopCopy(hailFile + ".bed", localHailFile + ".bed", hadoopConf)
    hadoopCopy(hailFile + ".bim", localHailFile + ".bim", hadoopConf)
    hadoopCopy(hailFile + ".fam", localHailFile + ".fam", hadoopConf)

    // use plink to assert that the concordance rate is 1
    val exitCode = s"plink --bfile ${ uriPath(localBFile) } --bmerge ${ uriPath(localHailFile) } --merge-mode 6 --out ${ uriPath(localMergeFile) }" !

    // assert that plink exited successfully
    assert(exitCode == 0)

    // assert that the .diff file is empty of non-header columns
    assert(
      readFile(localMergeFile + ".diff", sc.hadoopConfiguration) { is =>
        Source.fromInputStream(is)
          .getLines()
          .toIndexedSeq
          .map(_.split(" +").filter(!_.isEmpty).toIndexedSeq) == IndexedSeq(IndexedSeq("SNP", "FID", "IID", "NEW", "OLD"))
      }
    )
  }
}
