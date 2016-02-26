package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.{ImportPlinkBfile, ExportPlink, State, SplitMulti}
import org.broadinstitute.hail.io.LoadVCF
import org.testng.annotations.Test
import sys.process._

import scala.io.Source

class ImportPlinkSuite extends SparkSuite {
  def rewriteBimIDs(file: String) {
    val parsed = readFile(file, sc.hadoopConfiguration) { is =>
      Source.fromInputStream(is)
        .getLines()
        .toList
        .map(line => line
          .split("\t"))
        .map(arr =>
          s"${arr(0)}\t${s"${arr(0)}:${arr(3)}:${arr(5)}:${arr(4)}"}\t${arr(2)}\t${arr(3)}\t${arr(4)}\t${arr(5)}")
    }
    writeTable(file, sc.hadoopConfiguration, parsed)
  }

  @Test def testImportIdenticalToExport() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    val exportState1 = ExportPlink.run(state, Array("-o", "/tmp/hailOut"))

    val importState1 = ImportPlinkBfile.run(state, Array("--bfile","/tmp/hailOut"))
    val splitState1 = SplitMulti.run(importState1, Array.empty[String])
    val exportState2 = ExportPlink.run(splitState1, Array("-o", "/tmp/hailOut2"))
    val importState2 = ImportPlinkBfile.run(state, Array("--bfile","/tmp/hailOut2"))

    assert(importState1.vds.same(importState2.vds))

    val exitCodeFam = "diff /tmp/hailOut.fam /tmp/hailOut2.fam" !
    val exitCodeBim = "diff /tmp/hailOut.bim /tmp/hailOut2.bim" !
    val exitCodeBed = "diff /tmp/hailOut.bed /tmp/hailOut2.bed" !

    assert(exitCodeFam == 0 && exitCodeBim == 0 && exitCodeBed == 0)

  }
}
