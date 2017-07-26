package is.hail.io

import java.io.{File, FileWriter}

import is.hail.SparkSuite
import is.hail.expr.TStruct
import is.hail.io.vcf.{GenericRecordReader, LoadGDB, LoadVCF}
import is.hail.variant.GenericDataset
import org.testng.annotations.Test

class ImportGDBSuite extends SparkSuite {

  val loader = "/Users/jgoldsmi/tdbjson/loader_sample2.json"
  val workspace = "/Users/jgoldsmi/tdbworkspace"
  val arrName = "my_array_sample2"
  val ref = "/Users/jgoldsmi/build/Homo_sapiens_assembly19.fasta"
  val reader = new GenericRecordReader(Set.empty)
  val gdbVariantSampleMatrix: GenericDataset = LoadGDB(hc, reader, loader, workspace, arrName, ref) //TODO: hc.importGDB(loader, workspace, arrName, ref) @HailContext.scala
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

  //FIXME: this test fails because of the differences in info signatures AND filters -- waiting on GenomicsDB response to figure out how to fix infoSignatures
  @Test def genomicsDBVariantAnnotationSignatures() {

    val vcfVAS = vcfVariantSampleMatrix
      .metadata
      .vaSignature
      .asInstanceOf[TStruct]
      .fields
      .toSet

    val gdbVAS = gdbVariantSampleMatrix
      .metadata
      .vaSignature
      .asInstanceOf[TStruct]
      .fields
      .toSet

    assert(vcfVAS.equals(gdbVAS))
  }

  //TODO: this test maps each field to a pair of (name, type) because GenomicsDB drops descriptions for each field -- find way to get descriptions into GenomicsDB array
  @Test def genomicsDBGenotypeSignature() {

    val vcfGenotypeSignature = vcfVariantSampleMatrix
      .genotypeSignature
      .asInstanceOf[TStruct]
      .fields
      .map(field => {
        //println(field.toString + "\n")
        (field.name, field.typ)
      })
      .toSet

    val gdbGenotypeSignature = gdbVariantSampleMatrix
      .genotypeSignature
      .asInstanceOf[TStruct]
      .fields
      .map(field => {
        //println(field.toString + "\n")
        (field.name, field.typ)
      })
      .toSet

    assert(vcfGenotypeSignature.equals(gdbGenotypeSignature))
  }

  @Test def genomicsDBVariantsSamples() {

    assert(vcfVariantSampleMatrix.count == gdbVariantSampleMatrix.count) //(numSamples, numVariants)

    assert(vcfVariantSampleMatrix.variants.collect.sameElements(gdbVariantSampleMatrix.variants.collect))

    assert(vcfVariantSampleMatrix.sampleIds.equals(gdbVariantSampleMatrix.sampleIds))
  }

  //FIXME: this test fails because GT field not getting initialized (despite turning on produce_GT_field option in loader file)
  @Test def genomicsDBRDDGenotypeData() {

    val vcfGT = vcfVariantSampleMatrix.rdd.map(_._2._2).collect
    val gdbGT = gdbVariantSampleMatrix.rdd.map(_._2._2).collect

    val writer = new FileWriter(new File("/Users/jgoldsmi/Desktop/gtInfo.txt"))


    assert(vcfGT.zip(gdbGT).forall( { case (it1, it2) => it1.zip(it2).forall( { case (a1, a2) =>
      writer.write(a1.toString + "\n")
      writer.write(a2.toString + "\n\n")
      a1.equals(a2) })
    }))

    writer.close()
  }

  //FIXME: this test fails because the info signatures are different due to gdb issues, and the QUAL field is dropped
  @Test def genomicsDBRDDAnnotations() {

    val vcfAnn = vcfVariantSampleMatrix.rdd.map(_._2._1).collect
    val gdbAnn = gdbVariantSampleMatrix.rdd.map(_._2._1).collect

    assert(vcfAnn.zip(gdbAnn).forall( { case (a1, a2) =>
      //println(a1.toString)
      //println(a2.toString)
      //println
      a1.equals(a2)
    }))
  }

  @Test def genomicsDBSampleAnnotations() {

    val vcfSA = vcfVariantSampleMatrix.sampleAnnotations
    val gdbSA = gdbVariantSampleMatrix.sampleAnnotations

    assert(vcfSA.equals(gdbSA))
  }
}
