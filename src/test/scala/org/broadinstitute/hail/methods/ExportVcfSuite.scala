package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class ExportVcfSuite extends SparkSuite {
  @Test def test() {
    val vcfFile = "src/test/resources/sample.vcf"
    val tmpDir = "/tmp/"
    val outFile = tmpDir + "testExportVcf.vcf"

    val vdsOrig = LoadVCF(sc,vcfFile)
    val stateOrig = State("", sc, sqlContext, vdsOrig)
    ExportVCF.run(stateOrig, Array("-o",outFile,"-t",tmpDir))

    val vdsNew = LoadVCF(sc,outFile)
    val stateNew = State("", sc, sqlContext, vdsNew)

    assert(stateOrig.vds.same(stateNew.vds))

  }
}
