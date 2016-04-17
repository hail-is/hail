package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._
import org.testng.annotations.Test
import scala.io.Source
import sys.process._
import scala.language._

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

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    vds.rdd.count
    ExportPlink.run(state, Array("-o", "/tmp/hailOut"))

    // use plink to convert sample.vcf to a bed/bim/fam file
    s"plink --vcf src/test/resources/sample.vcf --make-bed --out /tmp/plinkOut --const-fid --keep-allele-order" !

    rewriteBimIDs("/tmp/hailOut.bim")
    rewriteBimIDs("/tmp/plinkOut.bim")

    // use plink to assert that the concordance rate is 1
    val exitCode = s"plink --bfile /tmp/plinkOut --bmerge /tmp/hailOut --merge-mode 6 --out /tmp/plinkHailMerge" !

    hadoopDelete("/tmp/plinkOut.bed", sc.hadoopConfiguration, recursive = true)
    hadoopDelete("/tmp/plinkOut.bim", sc.hadoopConfiguration, recursive = true)
    hadoopDelete("/tmp/plinkOut.fam", sc.hadoopConfiguration, recursive = true)

    hadoopDelete("/tmp/hailOut.bed", sc.hadoopConfiguration, recursive = true)
    hadoopDelete("/tmp/hailOut.bim", sc.hadoopConfiguration, recursive = true)
    hadoopDelete("/tmp/hailOut.fam", sc.hadoopConfiguration, recursive = true)

    // assert that plink exited successfully
    assert(exitCode == 0)

    // assert that the .diff file is empty of non-header columns
    assert(
      readFile("/tmp/plinkHailMerge.diff", sc.hadoopConfiguration) { is =>
        Source.fromInputStream(is)
          .getLines()
          .toIndexedSeq
          .map(_.split(" +").filter(!_.isEmpty).toIndexedSeq) == IndexedSeq(IndexedSeq("SNP", "FID", "IID", "NEW", "OLD"))
      }
    )
  }
}
