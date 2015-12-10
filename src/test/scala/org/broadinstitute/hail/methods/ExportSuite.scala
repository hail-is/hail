package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

import scala.io.Source

/**
  * This testing suite evaluates the [[org.broadinstitute.hail.driver.ExportVariants]]
  * and [[org.broadinstitute.hail.driver.ExportSamples]] commands, and verifies that
  * their output agrees with [[org.broadinstitute.hail.driver.VariantQC]] and
  * [[org.broadinstitute.hail.driver.SampleQC]] commands.
  */
class ExportSuite extends SparkSuite{

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

    SampleQC.run(state, Array("-o", "src/test/resources/sample.vcf.sampleQC"))
    val postSampleQC = SampleQC.run(state, Array("--store"))

    ExportSamples.run(postSampleQC, Array("-o", "src/test/resources/sample.vcf.exportSamples", "-c",
      "s.id, sa.qc.nCalled,sa.qc.nNotCalled,sa.qc.nHomRef,sa.qc.nHet,sa.qc.nHomVar,sa.qc.nSNP,sa.qc.nInsertion," +
        "sa.qc.nDeletion,sa.qc.nSingleton,sa.qc.nTransition,sa.qc.nTransversion,sa.qc.dpMean,sa.qc.dpStDev," +
        "sa.qc.dpMeanHomRef,sa.qc.dpStDevHomRef,sa.qc.dpMeanHet,sa.qc.dpStDevHet,sa.qc.dpMeanHomVar," +
        "sa.qc.dpStDevHomVar,sa.qc.gqMean,sa.qc.gqStDev,sa.qc.gqMeanHomRef,sa.qc.gqStDevHomRef,sa.qc.gqMeanHet," +
        "sa.qc.gqStDevHet,sa.qc.gqMeanHomVar,sa.qc.gqStDevHomVar,sa.qc.nNonRef,sa.qc.rTiTv,sa.qc.rHetHomVar," +
        "sa.qc.rDeletionInsertion"))

    val sQcOutput = Source.fromFile("src/test/resources/sample.vcf.sampleQC/part-00000")
      .getLines().toSet
    val sExportOutput = Source.fromFile("src/test/resources/sample.vcf.exportSamples/part-00000")
      .getLines().toSet

    assert(sQcOutput == sExportOutput)

    VariantQC.run(state, Array("-o", "src/test/resources/sample.vcf.variantQC"))

    val postVariantQC = VariantQC.run(state, Array("--store"))

    ExportVariants.run(postVariantQC, Array("-o", "src/test/resources/sample.vcf.exportVariants", "-c",
      "v.contig,v.start,v.ref,v.alt,va.qc.nCalled,va.qc.nNotCalled,va.qc.nHomRef,va.qc.nHet,va.qc.nHomVar,va.qc.dpMean,va.qc.dpStDev," +
        "va.qc.dpMeanHomRef,va.qc.dpStDevHomRef,va.qc.dpMeanHet,va.qc.dpStDevHet,va.qc.dpMeanHomVar," +
        "va.qc.dpStDevHomVar,va.qc.gqMean,va.qc.gqStDev,va.qc.gqMeanHomRef,va.qc.gqStDevHomRef," +
        "va.qc.gqMeanHet,va.qc.gqStDevHet,va.qc.gqMeanHomVar,va.qc.gqStDevHomVar,va.qc.MAF,va.qc.nNonRef," +
        "va.qc.rHeterozygosity,va.qc.rHetHomVar,va.qc.rExpectedHetFrequency,va.qc.pHWE"))

    val vQcOutput = Source.fromFile("src/test/resources/sample.vcf.variantQC/part-00000")
      .getLines().toSet
    val vExportOutput = Source.fromFile("src/test/resources/sample.vcf.exportVariants/part-00000")
      .getLines().toSet

    assert(vQcOutput == vExportOutput)
  }
}
