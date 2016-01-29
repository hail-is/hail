package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.testng.annotations.Test

class AnnotateFromFilesSuite extends SparkSuite {

  @Test def testSampleTSV {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    println(vds.nSamples)

    val state = State(sc, sqlContext, vds)

    val anno1 = AnnotateSamples.run(state, Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-t", "qPhen=Int"))
    val filt1 = FilterSamples.run(anno1, Array("--keep", "-c", "sa.Status == \"CASE\""))
    println(filt1.vds.nLocalSamples)

    val anno2 = AnnotateSamples.run(state, Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-n", "phenotype"))
    val filt2 = FilterSamples.run(anno2, Array("--keep", "-c", "sa.phenotype.Status == \"CASE\""))
    println(filt2.vds.nLocalSamples)

  }

  @Test def testVariantTSV() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    println(vds.nVariants)

    val state = State(sc, sqlContext, vds)

    val anno1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "stuff"))
    val filt1 = FilterVariants.run(anno1, Array("--keep", "-c", "va.stuff.Rand1 < .9"))
    println(filt1.vds.nVariants)
  }

  @Test def testVCF() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State(sc, sqlContext, vds)
    println(vds.nVariants)

    val anno1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/sampleInfoOnly.vcf", "--root", "other"))
    val filt1 = FilterVariants.run(anno1, Array("--keep", "-c", "va.other.pass"))
    filt1.vds.variantsAndAnnotations.collect().foreach { case (v, va) => println(v, va)}
    println(filt1.vds.nVariants)

  }
}
