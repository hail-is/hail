package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.Utils._
import org.testng.annotations.Test
import scala.io.Source

class ExportSuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State(sc, sqlContext, vds)

    SampleQC.run(state, Array("-o", "/tmp/sampleQC.tsv"))
    val postSampleQC = SampleQC.run(state, Array("--store"))

    assert(toTSVString(Array(1, 2, 3, 4, 5)) == "1,2,3,4,5")
    assert(toTSVString(5.124) == "5.1240e+00")

    ExportSamples.run(postSampleQC, Array("-o", "/tmp/exportSamples.tsv", "-c",
      "sampleID=s.id, callRate=sa.qc.callRate,nCalled=sa.qc.nCalled,nNotCalled=sa.qc.nNotCalled,nHomRef=sa.qc.nHomRef," +
        "nHet=sa.qc.nHet,nHomVar=sa.qc.nHomVar,nSNP=sa.qc.nSNP,nInsertion=sa.qc.nInsertion," +
        "nDeletion=sa.qc.nDeletion,nSingleton=sa.qc.nSingleton,nTransition=sa.qc.nTransition," +
        "nTransversion=sa.qc.nTransversion,dpMean=sa.qc.dpMean,dpStDev=sa.qc.dpStDev," +
        "dpMeanHomRef=sa.qc.dpMeanHomRef,dpStDevHomRef=sa.qc.dpStDevHomRef,dpMeanHet=sa.qc.dpMeanHet,dpStDevHet=sa.qc.dpStDevHet," +
        "dpMeanHomVar=sa.qc.dpMeanHomVar,dpStDevHomVar=sa.qc.dpStDevHomVar,gqMean=sa.qc.gqMean,gqStDev=sa.qc.gqStDev," +
        "gqMeanHomRef=sa.qc.gqMeanHomRef,gqStDevHomRef=sa.qc.gqStDevHomRef,gqMeanHet=sa.qc.gqMeanHet," +
        "gqStDevHet=sa.qc.gqStDevHet,gqMeanHomVar=sa.qc.gqMeanHomVar,gqStDevHomVar=sa.qc.gqStDevHomVar,nNonRef=sa.qc.nNonRef," +
        "rTiTv=sa.qc.rTiTv,rHetHomVar=sa.qc.rHetHomVar," +
        "rDeletionInsertion=sa.qc.rDeletionInsertion"))


    val sQcOutput = Source.fromFile("/tmp/sampleQC.tsv")
      .getLines().toSet
    val sExportOutput = Source.fromFile("/tmp/exportSamples.tsv")
      .getLines().toSet

    assert(sQcOutput == sExportOutput)

    VariantQC.run(state, Array("-o", "/tmp/variantQC.tsv"))
    val postVariantQC = VariantQC.run(state, Array("--store"))

    ExportVariants.run(postVariantQC, Array("-o", "/tmp/exportVariants.tsv", "-c",
      "Chrom=v.contig,Pos=v.start,Ref=v.ref,Alt=v.alt,callRate=va.qc.callRate,MAC=va.qc.MAC,MAF=va.qc.MAF," +
        "nCalled=va.qc.nCalled,nNotCalled=va.qc.nNotCalled," +
        "nHomRef=va.qc.nHomRef,nHet=va.qc.nHet,nHomVar=va.qc.nHomVar,dpMean=va.qc.dpMean,dpStDev=va.qc.dpStDev," +
        "dpMeanHomRef=va.qc.dpMeanHomRef,dpStDevHomRef=va.qc.dpStDevHomRef,dpMeanHet=va.qc.dpMeanHet,dpStDevHet=va.qc.dpStDevHet," +
        "dpMeanHomVar=va.qc.dpMeanHomVar,dpStDevHomVar=va.qc.dpStDevHomVar,gqMean=va.qc.gqMean,gqStDev=va.qc.gqStDev," +
        "gqMeanHomRef=va.qc.gqMeanHomRef,gqStDevHomRef=va.qc.gqStDevHomRef," +
        "gqMeanHet=va.qc.gqMeanHet,gqStDevHet=va.qc.gqStDevHet,gqMeanHomVar=va.qc.gqMeanHomVar,gqStDevHomVar=va.qc.gqStDevHomVar," +
        "nNonRef=va.qc.nNonRef," +
        "rHeterozygosity=va.qc.rHeterozygosity,rHetHomVar=va.qc.rHetHomVar,rExpectedHetFrequency=va.qc.rExpectedHetFrequency," +
        "pHWE=va.qc.pHWE"))

    val vQcOutput = Source.fromFile("/tmp/variantQC.tsv")
      .getLines().toSet
    val vExportOutput = Source.fromFile("/tmp/exportVariants.tsv")
      .getLines().toSet

    assert(vQcOutput == vExportOutput)
  }
}
