package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

import scala.io.Source

class ExportSuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    var state = State(sc, sqlContext, vds)
    state = SplitMulti.run(state, Array.empty[String])

    val sampleQCFile = tmpDir.createTempFile("sampleqc", ".tsv")
    val exportSamplesFile = tmpDir.createTempFile("exportsamples", ".tsv")

    SampleQC.run(state, Array("-o", sampleQCFile))
    val postSampleQC = SampleQC.run(state, Array.empty[String])

    val sb = new StringBuilder()
    sb.tsvAppend(Array(1, 2, 3, 4, 5))
    assert(sb.result() == "1,2,3,4,5")

    sb.clear()
    sb.tsvAppend(5.124)
    assert(sb.result() == "5.1240e+00")

    ExportSamples.run(postSampleQC, Array("-o", exportSamplesFile, "-c",
      "Sample=s.id, callRate=sa.qc.callRate,nCalled=sa.qc.nCalled,nNotCalled=sa.qc.nNotCalled,nHomRef=sa.qc.nHomRef," +
        "nHet=sa.qc.nHet,nHomVar=sa.qc.nHomVar,nSNP=sa.qc.nSNP,nInsertion=sa.qc.nInsertion," +
        "nDeletion=sa.qc.nDeletion,nSingleton=sa.qc.nSingleton,nTransition=sa.qc.nTransition," +
        "nTransversion=sa.qc.nTransversion,dpMean=sa.qc.dpMean,dpStDev=sa.qc.dpStDev," +
        "gqMean=sa.qc.gqMean,gqStDev=sa.qc.gqStDev," +
        "nNonRef=sa.qc.nNonRef," +
        "rTiTv=sa.qc.rTiTv,rHetHomVar=sa.qc.rHetHomVar," +
        "rDeletionInsertion=sa.qc.rDeletionInsertion"))

    val sQcOutput = readFile(sampleQCFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines().toSet
    }
    val sExportOutput = readFile(exportSamplesFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines().toSet
    }

    assert(sQcOutput == sExportOutput)

    val variantQCFile = tmpDir.createTempFile("variantqc", ".tsv")
    val exportVariantsFile = tmpDir.createTempFile("exportvariants", ".tsv")

    VariantQC.run(state, Array("-o", variantQCFile))
    val postVariantQC = VariantQC.run(state, Array.empty[String])

    ExportVariants.run(postVariantQC, Array("-o", exportVariantsFile, "-c",
      "Chrom=v.contig,Pos=v.start,Ref=v.ref,Alt=v.alt,callRate=va.qc.callRate,AC=va.qc.AC,AF=va.qc.AF," +
        "nCalled=va.qc.nCalled,nNotCalled=va.qc.nNotCalled," +
        "nHomRef=va.qc.nHomRef,nHet=va.qc.nHet,nHomVar=va.qc.nHomVar,dpMean=va.qc.dpMean,dpStDev=va.qc.dpStDev," +
        "gqMean=va.qc.gqMean,gqStDev=va.qc.gqStDev," +
        "nNonRef=va.qc.nNonRef," +
        "rHeterozygosity=va.qc.rHeterozygosity,rHetHomVar=va.qc.rHetHomVar,rExpectedHetFrequency=va.qc.rExpectedHetFrequency," +
        "pHWE=va.qc.pHWE"))

    val vQcOutput = readFile(variantQCFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines().toSet
    }
    val vExportOutput = readFile(exportVariantsFile, hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines().toSet
    }

    assert(vQcOutput == vExportOutput)
  }

  @Test def testExportSamples() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = FilterSamplesExpr.run(s, Array("--keep", "-c", """s.id == "C469::HG02026""""))
    assert(s.vds.nSamples == 1)

    // verify exports localSamples
    val f = tmpDir.createTempFile("samples", ".tsv")
    s = ExportSamples.run(s, Array("-o", f, "-c", "s.id"))
    assert(sc.textFile(f).count() == 1)
  }

  @Test def testAllowedNames() {
    var s = State(sc, sqlContext)

    val f = tmpDir.createTempFile("samples", ".tsv")
    val f2 = tmpDir.createTempFile("samples", ".tsv")
    val f3 = tmpDir.createTempFile("samples", ".tsv")

    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = ExportSamples.run(s, Array("-o", f, "-c", "S.A.M.P.L.E.ID = s.id"))
    s = ExportSamples.run(s, Array("-o", f2, "-c",
      "$$$YO_DAWG_I_HEARD_YOU_LIKE_%%%_#@!_WEIRD_CHARS**** = s.id, ANOTHERTHING=s.id"))
    s = ExportSamples.run(s, Array("-o", f3, "-c",
      "`I have some spaces and tabs\there` = s.id,`more weird stuff here`=s.id"))
    readFile(f, sc.hadoopConfiguration) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "S.A.M.P.L.E.ID")
    }
    readFile(f2, sc.hadoopConfiguration) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "$$$YO_DAWG_I_HEARD_YOU_LIKE_%%%_#@!_WEIRD_CHARS****\tANOTHERTHING")
    }
    readFile(f3, sc.hadoopConfiguration) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "I have some spaces and tabs\there\tmore weird stuff here")
    }
  }

  @Test def testIf() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))
    s = SplitMulti.run(s, Array.empty[String])
    s = SampleQC.run(s, Array.empty[String])

    // this should run without errors
    val f = tmpDir.createTempFile("samples", ".tsv")
    s = ExportSamples.run(s, Array("-o", f, "-c", "computation = 5 * (if (sa.qc.callRate < .95) 0 else 1)"))
  }
}
