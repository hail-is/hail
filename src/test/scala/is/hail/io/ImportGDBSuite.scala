package is.hail.io

import is.hail.SparkSuite
import is.hail.expr.TStruct
import is.hail.io.vcf.{GenericRecordReader, LoadGDB, LoadVCF}
import is.hail.variant.GenericDataset
import org.testng.annotations.Test

class ImportGDBSuite extends SparkSuite {

  val loader = "/Users/jgoldsmi/tdbjson/sample2loader.json"
  val query = "/Users/jgoldsmi/tdbjson/sample2query.json"
  val workspace = "/Users/jgoldsmi/tdbworkspace"
  val arrName = "my_array_sample2"
  val ref = "/Users/jgoldsmi/build/Homo_sapiens_assembly19.fasta"
  val reader = new GenericRecordReader(Set.empty)
  val gdbVariantSampleMatrix: GenericDataset = LoadGDB(hc, reader, loader, query, workspace, arrName, ref)
  val vcfVariantSampleMatrix: GenericDataset = LoadVCF(hc, reader, "src/test/resources/sample2.vcf")

  @Test def genomicsDBIterator() {

    import com.intel.genomicsdb.GenomicsDBFeatureReader

    val numVariants = hc
      .importVCF("src/test/resources/sample2.vcf")
      .countVariants()

    val codec = new htsjdk.variant.vcf.VCFCodec()
    val gdbIterator = new GenomicsDBFeatureReader(loader, workspace, arrName, ref, codec).iterator

    var count = 0
    while (gdbIterator.hasNext) {
      val record = gdbIterator.next
      if (!record.getAlternateAlleles.get(0).toString.equals("*")) {
        count += 1
      }
    }

    gdbIterator.close()

    assert(count==numVariants)
  }

  @Test def genomicsDBGlobalAnnotation() {

    val vcfGA = vcfVariantSampleMatrix.globalAnnotation
    val gdbGA = gdbVariantSampleMatrix.globalAnnotation

    assert(vcfGA == gdbGA)
  }

  @Test def genomicsDBIsGenericGenotype() {

    val vcfIsGenericGenotype = vcfVariantSampleMatrix.metadata.isGenericGenotype
    val gdbIsGenericGenotype = gdbVariantSampleMatrix.metadata.isGenericGenotype

    assert(vcfIsGenericGenotype == gdbIsGenericGenotype)
  }

  @Test def genomicsDBNoMulti() {

    assert(vcfVariantSampleMatrix.wasSplit == gdbVariantSampleMatrix.wasSplit) // wasSplit == noMulti
  }

  @Test def genomicsDBVariantAnnotationSignatures() {

    val vcfVAS = vcfVariantSampleMatrix
      .metadata
      .vaSignature
      .asInstanceOf[TStruct]
      .fields

    val gdbVAS = gdbVariantSampleMatrix
      .metadata
      .vaSignature
      .asInstanceOf[TStruct]
      .fields

    assert(vcfVAS.zip(gdbVAS).forall( { case (f1, f2) =>
        println(f1)
        println(f2)
        println
        f1.equals(f2)
    }))
  }

  @Test def genomicsDBGenotypeSignature() {

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

  @Test def genomicsDBVariantsSamples() {

    assert(vcfVariantSampleMatrix.count == gdbVariantSampleMatrix.count) //(numSamples, numVariants)

    assert(vcfVariantSampleMatrix.variants.collect.sameElements(gdbVariantSampleMatrix.variants.collect))

    assert(vcfVariantSampleMatrix.sampleIds.equals(gdbVariantSampleMatrix.sampleIds))
  }

  @Test def genomicsDBRDDGenotypeData() {

    val vcfGT = vcfVariantSampleMatrix
      .rdd
      .map(_._2._2)
      .collect

    val gdbGT = gdbVariantSampleMatrix
      .rdd
      .map(_._2._2)
      .collect

    assert(vcfGT.zip(gdbGT).forall( { case (it1, it2) => it1.zip(it2).forall( { case (a1, a2) => a1.equals(a2) }) }))
  }

  //FIXME: this test fails because genomicsDB drops the QUAL field, and the DP field has two extra zeroes
  @Test def genomicsDBRDDAnnotations() {

    val vcfAnn = vcfVariantSampleMatrix
      .rdd
      .map(_._2._1.asInstanceOf[annotation.Annotation])
      .collect

    val gdbAnn = gdbVariantSampleMatrix
      .rdd
      .map(_._2._1.asInstanceOf[annotation.Annotation])
      .collect

    assert(vcfAnn.zip(gdbAnn).forall( { case (a1, a2) =>
      println(a1.toString)
      println(a2.toString)
      println
      a1.equals(a2)
    }))
  }

  @Test def genomicsDBSampleAnnotations() {

    val vcfSA = vcfVariantSampleMatrix.sampleAnnotations
    val gdbSA = gdbVariantSampleMatrix.sampleAnnotations

    assert(vcfSA.equals(gdbSA))
  }
}
