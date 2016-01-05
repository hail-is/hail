package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.methods.UserExportUtils.toTSVString
import org.broadinstitute.hail.variant.Sample
import org.testng.annotations.Test
import scala.io.Source

/**
  * This testing suite evaluates the [[org.broadinstitute.hail.driver.ExportVariants]]
  * and [[org.broadinstitute.hail.driver.ExportSamples]] commands, and verifies that
  * their output agrees with [[org.broadinstitute.hail.driver.VariantQC]] and
  * [[org.broadinstitute.hail.driver.SampleQC]] commands.
  */
class ExportSuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

    SampleQC.run(state, Array("-o", "/tmp/sampleQC"))
    val postSampleQC = SampleQC.run(state, Array("--store"))

    assert(toTSVString(Some(5.1)) == "5.1000e+00")
    assert(toTSVString(Some(None)) == "NA")
    assert(toTSVString(Array(1,2,3,4,5)) == "1,2,3,4,5")
    assert(toTSVString(5.124) == "5.1240e+00")

    ExportSamples.run(postSampleQC, Array("-o", "/tmp/exportSamples", "-c",
      "sample=s.id, callRate=sa.qc.callRate,nCalled=sa.qc.nCalled,nNotCalled=sa.qc.nNotCalled,nHomRef=sa.qc.nHomRef," +
        "nHet=sa.qc.nHet,nHomVar=sa.qc.nHomVar,nSNP=sa.qc.nSNP,nInsertion=sa.qc.nInsertion," +
        "nDeletion=sa.qc.nDeletion,nSingelton=sa.qc.nSingleton,nTransition=sa.qc.nTransition," +
        "nTransversion=sa.qc.nTransversion,dpM=sa.qc.dpMean,dpSD=sa.qc.dpStDev," +
        "dpMHR=sa.qc.dpMeanHomRef,dpSDHR=sa.qc.dpStDevHomRef,dpMH=sa.qc.dpMeanHet,dpSDH=sa.qc.dpStDevHet," +
        "dpMHV=sa.qc.dpMeanHomVar,DPSDHV=sa.qc.dpStDevHomVar,gqM=sa.qc.gqMean,GQSD=sa.qc.gqStDev," +
        "GQMHR=sa.qc.gqMeanHomRef,GQSDHR=sa.qc.gqStDevHomRef,GQMH=sa.qc.gqMeanHet," +
        "GQSDH=sa.qc.gqStDevHet,GQMHV=sa.qc.gqMeanHomVar,GQSDHV=sa.qc.gqStDevHomVar,nNonRef=sa.qc.nNonRef," +
        "rTiTv=sa.qc.rTiTv,rHetHomVar=sa.qc.rHetHomVar," +
        "rDelIns=sa.qc.rDeletionInsertion"))


    val sQcOutput = Source.fromFile("/tmp/sampleQC/part-00000")
      .getLines().toSet
    val sExportOutput = Source.fromFile("/tmp/exportSamples/part-00000")
      .getLines().toSet

    assert(sQcOutput == sExportOutput)

    VariantQC.run(state, Array("-o", "/tmp/variantQC"))
    val postVariantQC = VariantQC.run(state, Array("--store"))

    ExportVariants.run(postVariantQC, Array("-o", "/tmp/exportVariants", "-c",
      "chr=v.contig,pos=v.start,ref=v.ref,alt=v.alt,callRate=va.qc.callRate,MAC=va.qc.MAC,MAF=va.qc.MAF," +
        "nCalled=va.qc.nCalled,nNotCalled=va.qc.nNotCalled," +
        "nHomRef=va.qc.nHomRef,nHet=va.qc.nHet,nHomVar=va.qc.nHomVar,dpM=va.qc.dpMean,dpSD=va.qc.dpStDev," +
        "dpMHR=va.qc.dpMeanHomRef,dpSDHR=va.qc.dpStDevHomRef,dpMH=va.qc.dpMeanHet,dpSDH=va.qc.dpStDevHet," +
        "dpMHV=va.qc.dpMeanHomVar,dpSDHV=va.qc.dpStDevHomVar,gqM=va.qc.gqMean,gqSD=va.qc.gqStDev," +
        "gqMHR=va.qc.gqMeanHomRef,gqSDHR=va.qc.gqStDevHomRef," +
        "gqMH=va.qc.gqMeanHet,gqSDH=va.qc.gqStDevHet,gqMHV=va.qc.gqMeanHomVar,gqSDHV=va.qc.gqStDevHomVar," +
        "nNonRef=va.qc.nNonRef," +
        "rHet=va.qc.rHeterozygosity,rHetHomVar=va.qc.rHetHomVar,rExpHetFreq=va.qc.rExpectedHetFrequency," +
        "pHWE=va.qc.pHWE"))


    val vQcOutput = Source.fromFile("/tmp/variantQC/part-00000")
      .getLines().toSet
    val vExportOutput = Source.fromFile("/tmp/exportVariants/part-00000")
      .getLines().toSet

    assert(vQcOutput == vExportOutput)

    val (headers, exprs) = ExportTSV.parseExpression("maf=5, foobar=va.qc.FOOBAR")
  }
}
