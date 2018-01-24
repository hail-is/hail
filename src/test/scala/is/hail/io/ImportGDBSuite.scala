package is.hail.io

import is.hail.SparkSuite
import is.hail.expr.types.TStruct
import is.hail.io.vcf.{HtsjdkRecordReader, LoadGDB, LoadVCF}
import is.hail.testUtils._
import org.testng.annotations.Test

class ImportGDBSuite extends SparkSuite {
  /*TODO: when genomicsDB allows tiledb workspaces to move from creation location:
      -enable all tests
      -create new tiledb workspace with newest genomicsDB updates
      -replace old tiledb workspace in src/test/resources/ with new one
   */

  val loader = "src/test/resources/sample2loader.json"
  val workspace = "src/test/resources/tdbworkspace"
  val arrName = "sample2Array"
  val vid = "src/test/resources/sample2vid.json"
  val callsets = "src/test/resources/sample2callsets.json"
  val vcfHeader = Some("src/test/resources/sample2header.vcf")
  val vcf = "src/test/resources/sample2.vcf"

  @Test def genomicsDBGlobalAnnotation() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfGA = vcfVariantSampleMatrix.globalAnnotation
    val gdbGA = gdbVariantSampleMatrix.globalAnnotation

    assert(vcfGA == gdbGA)
  }

  @Test def genomicsDBVariantAnnotationSignatures() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfVAS = vcfVariantSampleMatrix
      .vaSignature

    val gdbVAS = gdbVariantSampleMatrix
      .vaSignature

    assert(vcfVAS.equals(gdbVAS))
  }

  @Test def genomicsDBGenotypeSignature() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfGenotypeSignature = vcfVariantSampleMatrix
      .genotypeSignature
      .asInstanceOf[TStruct]
      .fields

    val gdbGenotypeSignature = gdbVariantSampleMatrix
      .genotypeSignature
      .asInstanceOf[TStruct]
      .fields

    assert(vcfGenotypeSignature.zip(gdbGenotypeSignature).forall( { case (f1, f2) => f1.equals(f2) }))
  }

  @Test(enabled = false) def genomicsDBVariantsSamples() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    assert(vcfVariantSampleMatrix.count == gdbVariantSampleMatrix.count) //(numSamples, numVariants)

    assert(vcfVariantSampleMatrix.variants.collect.sameElements(gdbVariantSampleMatrix.variants.collect))

    assert(vcfVariantSampleMatrix.sampleIds.equals(gdbVariantSampleMatrix.sampleIds))
  }

  @Test def genomicsDBRDDGenotypeData() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfGT = vcfVariantSampleMatrix
      .rdd
      .map { case (_, (_, gt)) => gt }
      .collect

    val gdbGT = gdbVariantSampleMatrix
      .rdd
      .map { case (_, (_, gt)) => gt }
      .collect

    assert(vcfGT.zip(gdbGT).forall( { case (it1, it2) => it1.zip(it2).forall( { case (a1, a2) => a1.equals(a2) }) }))
  }

  //FIXME - genomicsDB bugs relating to this test: some FP numbers are off by a little, arrays (length "A") dropping all but first element
  @Test def genomicsDBRDDAnnotations() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfVAInfo = vcfVariantSampleMatrix.queryVA("va.info.AN")._2 //TODO: remove ".AN" once above issues are fixed
    val vcfAnn = vcfVariantSampleMatrix
      .rdd
      .map { case (_, (va, _)) => vcfVAInfo(va) }
      .collect

    val gdbVAInfo = gdbVariantSampleMatrix.queryVA("va.info.AN")._2
    val gdbAnn = gdbVariantSampleMatrix
      .rdd
      .map { case (_, (va, _)) => gdbVAInfo(va) }
      .collect

    assert(vcfAnn.zip(gdbAnn).forall( { case (a1, a2) => a1.equals(a2) }))
  }

  @Test(enabled = false) def genomicsDBSampleAnnotations() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfSA = vcfVariantSampleMatrix.sampleAnnotations
    val gdbSA = gdbVariantSampleMatrix.sampleAnnotations

    assert(vcfSA.equals(gdbSA))
  }

  @Test(enabled = false) def genomicsDBFilters() {
    val reader = new HtsjdkRecordReader(Set.empty)
    val gdbVariantSampleMatrix = LoadGDB(hc, reader, loader, workspace, arrName, vid, callsets, vcfHeader)
    val vcfVariantSampleMatrix = LoadVCF(hc, reader, Some(vcf), Array(vcf))

    val vcfQuery = vcfVariantSampleMatrix.queryVA("va.filters")._2
    val vcfFilters = vcfVariantSampleMatrix
      .rdd
      .map { case (_, (va, _)) => vcfQuery(va) }
      .collect

    val gdbQuery = gdbVariantSampleMatrix.queryVA("va.filters")._2
    val gdbFilters = gdbVariantSampleMatrix
      .rdd
      .map { case (_, (va, _)) => gdbQuery(va) }
      .collect                                      

    assert(vcfFilters.sameElements(gdbFilters))
  }
}
