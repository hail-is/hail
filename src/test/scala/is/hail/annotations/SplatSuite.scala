package is.hail.annotations

import is.hail.SparkSuite
import is.hail.methods.VariantQC
import is.hail.utils._
import org.testng.annotations.Test

class SplatSuite extends SparkSuite {
  @Test def test() {
    val out1 = tmpDir.createLocalTempFile("out1", ".txt")
    val out2 = tmpDir.createLocalTempFile("out2", ".txt")
    val out3 = tmpDir.createLocalTempFile("out3", ".txt")
    val out4 = tmpDir.createLocalTempFile("out4", ".txt")

    val vkt = VariantQC(hc.importVCF("src/test/resources/sample2.vcf")
      .splitMulti())
      .variantsKT()

    vkt.select("variant = v", "va.qc.callRate", "va.qc.AC", "va.qc.AF", "va.qc.nCalled", "va.qc.nNotCalled",
      "va.qc.nHomRef", "va.qc.nHet", "va.qc.nHomVar", "va.qc.dpMean", "va.qc.dpStDev",
      "va.qc.gqMean", "va.qc.gqStDev", "va.qc.nNonRef", "va.qc.rHeterozygosity", "va.qc.rHetHomVar",
      "va.qc.rExpectedHetFrequency", "va.qc.pHWE")
      .export(out1)

    vkt.select("variant = v", "va.qc.*").export(out2)

    assert(hadoopConf.readLines(out1)(_.map(_.value).toIndexedSeq) ==
      hadoopConf.readLines(out2)(_.map(_.value).toIndexedSeq))

    vkt.select("variant = v", "QC.callRate = va.qc.callRate", "QC.AC = va.qc.AC", "QC.AF = va.qc.AF",
    "QC.nCalled = va.qc.nCalled", "QC.nNotCalled=va.qc.nNotCalled", "QC.nHomRef=va.qc.nHomRef",
    "QC.nHet=va.qc.nHet", "QC.nHomVar=va.qc.nHomVar", "QC.dpMean=va.qc.dpMean", "QC.dpStDev=va.qc.dpStDev",
    "QC.gqMean=va.qc.gqMean", "QC.gqStDev=va.qc.gqStDev", "QC.nNonRef=va.qc.nNonRef", "QC.rHeterozygosity=va.qc.rHeterozygosity",
    "QC.rHetHomVar=va.qc.rHetHomVar", "QC.rExpectedHetFrequency=va.qc.rExpectedHetFrequency", "QC.pHWE=va.qc.pHWE")
      .export(out3)

    vkt.select("variant = v", "QC = va.qc.*").export(out4)

    assert(hadoopConf.readLines(out3)(_.map(_.value).toIndexedSeq) ==
      hadoopConf.readLines(out4)(_.map(_.value).toIndexedSeq))
  }
}
