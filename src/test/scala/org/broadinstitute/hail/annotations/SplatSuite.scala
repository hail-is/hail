package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.{SparkSuite, TestUtils}
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test
import org.broadinstitute.hail.utils._

class SplatSuite extends SparkSuite {
  @Test def test() {
    val out1 = tmpDir.createLocalTempFile("out1", ".txt")
    val out2 = tmpDir.createLocalTempFile("out2", ".txt")
    val out3 = tmpDir.createLocalTempFile("out3", ".txt")
    val out4 = tmpDir.createLocalTempFile("out4", ".txt")

    var s = ImportVCF.run(State(sc, sqlContext, null), "src/test/resources/sample2.vcf")
    s = SplitMulti.run(s)
    s = VariantQC.run(s)
    ExportVariants.run(s, "-o", out1, "-c",
      s"""variant = v, callRate=va.qc.callRate,AC=va.qc.AC,AF=va.qc.AF,
          |nCalled=va.qc.nCalled,nNotCalled=va.qc.nNotCalled,nHomRef=va.qc.nHomRef,
          |nHet=va.qc.nHet,nHomVar=va.qc.nHomVar,dpMean=va.qc.dpMean,dpStDev=va.qc.dpStDev,
          |gqMean=va.qc.gqMean,gqStDev=va.qc.gqStDev,nNonRef=va.qc.nNonRef,
          |rHeterozygosity=va.qc.rHeterozygosity,rHetHomVar=va.qc.rHetHomVar,
          |rExpectedHetFrequency=va.qc.rExpectedHetFrequency,pHWE=va.qc.pHWE""".stripMargin)
    ExportVariants.run(s, "-o", out2, "-c", "variant = v, va.qc.*")

    assert(hadoopConf.readLines(out1)(_.map(_.value).toIndexedSeq) ==
      hadoopConf.readLines(out2)(_.map(_.value).toIndexedSeq))

    ExportVariants.run(s, "-o", out3, "-c",
      s"""variant = v, QC.callRate=va.qc.callRate,QC.AC=va.qc.AC,QC.AF=va.qc.AF,
          |QC.nCalled=va.qc.nCalled,QC.nNotCalled=va.qc.nNotCalled,QC.nHomRef=va.qc.nHomRef,
          |QC.nHet=va.qc.nHet,QC.nHomVar=va.qc.nHomVar,QC.dpMean=va.qc.dpMean,QC.dpStDev=va.qc.dpStDev,
          |QC.gqMean=va.qc.gqMean,QC.gqStDev=va.qc.gqStDev,QC.nNonRef=va.qc.nNonRef,
          |QC.rHeterozygosity=va.qc.rHeterozygosity,QC.rHetHomVar=va.qc.rHetHomVar,
          |QC.rExpectedHetFrequency=va.qc.rExpectedHetFrequency,QC.pHWE=va.qc.pHWE""".stripMargin)
    ExportVariants.run(s, "-o", out4, "-c", "variant = v, QC = va.qc.*")

    assert(hadoopConf.readLines(out3)(_.map(_.value).toIndexedSeq) ==
      hadoopConf.readLines(out4)(_.map(_.value).toIndexedSeq))

    TestUtils.interceptFatal("unrealizable type.*Splat")(
      AnnotateVariants.run(s, "expr", "-c", "va = va.info.*"))
  }
}
