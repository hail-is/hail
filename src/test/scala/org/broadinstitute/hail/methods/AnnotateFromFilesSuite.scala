package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.{Gen, Properties}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.io.Source
import org.broadinstitute.hail.Utils._

class AnnotateFromFilesSuite extends SparkSuite {

  @Test def testSampleTSV {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    println(vds.nSamples)

    val state = State(sc, sqlContext, vds)

    val path1 = "src/test/resources/sampleAnnotations.tsv"
    val anno1 = AnnotateSamples.run(state, Array("-c", path1, "-s", "Sample", "-t", "qPhen:Int"))
    val fileMap = Source.fromInputStream(hadoopOpen(path1, vds.sparkContext.hadoopConfiguration))
      .getLines()
      .filter(line => !line.startsWith("Sample"))
      .map(line => {
        val split = line.split("\t")
        val f3 = split(2) match {
          case "NA" => None
          case x => Some(x.toInt)
        }
        (split(0), (split(1), f3))
      })
      .toMap
    anno1.vds.metadata.sampleIds.zip(anno1.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) || ((fileMap(id)._1 == sa.get[String]("Status")) && (fileMap(id)._2 == sa.getOption[Int]("qPhen")))
      }


    val anno2 = AnnotateSamples.run(state,
      Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-r", "phenotype", "-t", "qPhen:Int"))
    anno2.vds.metadata.sampleIds.zip(anno2.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) ||
            ((fileMap(id)._1 == sa.get[Annotations]("phenotype").get[String]("Status")) &&
              (fileMap(id)._2 == sa.get[Annotations]("phenotype").getOption[Int]("qPhen")))
      }


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
    filt1.vds.variantsAndAnnotations.collect().foreach { case (v, va) => println(v, va) }
    println(filt1.vds.nVariants)

  }


  @Test def testAnnotateVariants() {
    object Spec extends Properties("testAnnotateVariants") {
      property("annotateVariants") =
        forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { (vds: VariantDataset) =>
          var s = State(sc, sqlContext, vds)
          s = SplitMulti.run(s, Array[String]())
          s.vds.mapWithAll((v: Variant, va: Annotations, _: Int, g: Genotype) =>
            !g.fakeRef || va.attrs("wasSplit").asInstanceOf[Boolean])
            .collect()
            .forall(identity)
        }
    }


  }
}
