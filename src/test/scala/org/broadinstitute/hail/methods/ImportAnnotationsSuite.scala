package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.io.annotators.Annotator
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.io.Source
import org.broadinstitute.hail.Utils._

class ImportAnnotationsSuite extends SparkSuite {

  @Test def testSampleTSVAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")

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
        (split(0), (Some(split(1)), f3))
      })
      .toMap
    anno1.vds.metadata.sampleIds.zip(anno1.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) ||
            ((fileMap(id)._1 == sa.getOption[String]("Status")) && (fileMap(id)._2 == sa.getOption[Int]("qPhen")))
      }


    val anno2 = AnnotateSamples.run(state,
      Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-r", "phenotype", "-t", "qPhen:Int"))
    anno2.vds.metadata.sampleIds.zip(anno2.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) ||
            ((fileMap(id)._1 == sa.get[Annotations]("phenotype").getOption[String]("Status")) &&
              (fileMap(id)._2 == sa.get[Annotations]("phenotype").getOption[Int]("qPhen")))
      }
  }

  @Test def testVariantTSVAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])

    val fileMap = Source.fromInputStream(hadoopOpen("src/test/resources/variantAnnotations.tsv",
      sc.hadoopConfiguration))
      .getLines()
      .filter(line => !line.startsWith("Chromosome"))
      .map(line => {
        val split = line.split("\t")
        val v = Variant(split(0), split(1).toInt, split(2), split(3))
        val rand1 = if (split(4) == "NA") None else Some(split(4).toDouble)
        val rand2 = if (split(5) == "NA") None else Some(split(5).toDouble)
        val gene = if (split(6) == "NA") None else Some(split(6))
        (v, (rand1, rand2, gene))
      })
      .toMap

    val anno1 = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "stuff"))
    anno1.vds.rdd
      .collect()
      .foreach {
        case (v, va, gs) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(rand1 == va.get[Annotations]("stuff").getOption[Double]("Rand1"))
          assert(rand2 == va.get[Annotations]("stuff").getOption[Double]("Rand2"))
          assert(gene == va.get[Annotations]("stuff").getOption[String]("Gene"))
      }

    val anno2 = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double"))
    anno2.vds.rdd
      .collect()
      .foreach {
        case (v, va, gs) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(rand1 == va.getOption[Double]("Rand1"))
          assert(rand2 == va.getOption[Double]("Rand2"))
          assert(gene == va.getOption[String]("Gene"))
      }

    val anno1alternate = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.alternateformat.tsv", "--vcolumns",
        "Chromosome:Position:Ref:Alt", "-t", "Rand1:Double,Rand2:Double", "-r", "stuff"))

    val anno2alternate = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.alternateformat.tsv", "--vcolumns",
        "Chromosome:Position:Ref:Alt", "-t", "Rand1:Double,Rand2:Double"))

    assert(anno1alternate.vds.same(anno1.vds))
    assert(anno2alternate.vds.same(anno2.vds))
  }

  @Test def testVCFAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])

    val anno1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/sampleInfoOnly.vcf", "--root", "other"))

    val initialMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    anno1.vds.eraseSplit.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          assert(va.contains("other") &&
            initialMap(v).attrs == (va.attrs - "other"))
      }
  }

  @Test def testBedIntervalAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = Cache.run(SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String]), Array.empty[String])

    val bed1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/example1.bed"))
    val bed1r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example1.bed", "-r", "bed"))
    val bed2 = AnnotateVariants.run(state, Array("-c", "src/test/resources/example2.bed"))
    val bed2r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example2.bed", "-r", "bed"))
    val bed3 = AnnotateVariants.run(state, Array("-c", "src/test/resources/example3.bed"))
    val bed3r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example3.bed", "-r", "bed"))

    val map1r = bed1r.vds.variantsAndAnnotations
      .collect()
      .toMap

    bed1.vds.variantsAndAnnotations
      .collect()
      .foreach {
        case (v, va) =>
          v.start <= 14000000 ||
            v.start >= 17000000 ||
            (va.contains("BedTest") &&
              va.getOption[Boolean]("BedTest") == map1r(v).get[Annotations]("bed").getOption[Boolean]("BedTest"))
      }

    val map2r = bed1r.vds.variantsAndAnnotations
      .collect()
      .toMap

    bed2.vds.variantsAndAnnotations
      .collect()
      .foreach {
        case (v, va) =>
          v.start <= 14000000 ||
            v.start >= 17000000 ||
            (va.contains("BedTest") &&
              va.get[String]("BedTest") == map1r(v).get[Annotations]("bed").get[String]("BedTest"))
      }

    assert(bed3.vds.same(bed2.vds))
    assert(bed3r.vds.same(bed2r.vds))

    val int1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation1.interval_list",
      "-i", "BedTest"))
    val int1r = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation1.interval_list",
      "-i", "BedTest", "-r", "bed"))
    val int2 = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation2.interval_list",
      "-i", "BedTest"))
    val int2r = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation2.interval_list",
      "-i", "BedTest", "-r", "bed"))

    assert(int1.vds.same(bed1.vds))
    assert(int1r.vds.same(bed1r.vds))
    assert(int2.vds.same(bed2.vds))
    assert(int2r.vds.same(bed2r.vds))
  }

  @Test def testSerializedAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])

    val tsv1 = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double"))
    val tsv1r = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "stuff"))

    val vcf1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/sampleInfoOnly.vcf", "--root", "other"))

    ConvertAnnotations.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double",
        "-o", "/tmp/variantAnnotationsTSV.ser"))
    ConvertAnnotations.run(state,
      Array("-c", "src/test/resources/sampleInfoOnly.vcf", "-o", "/tmp/variantAnnotationsVCF.ser"))

    val tsvSer1 = AnnotateVariants.run(state,
      Array("-c", "/tmp/variantAnnotationsTSV.ser"))

    val tsvSer1r = AnnotateVariants.run(state,
      Array("-c", "/tmp/variantAnnotationsTSV.ser", "-r", "stuff"))

    val vcfSer1 = AnnotateVariants.run(state,
      Array("-c", "/tmp/variantAnnotationsVCF.ser", "-r", "other"))

    assert(tsv1.vds.same(tsvSer1.vds))
    assert(tsv1r.vds.same(tsvSer1r.vds))
    assert(vcf1.vds.same(vcfSer1.vds))
  }

  @Test def testRootFunction() {
    val f1 = Annotator.rootFunction(null)
    val f2 = Annotator.rootFunction("info")
    val f3 = Annotator.rootFunction("other.info")
    val f4 = Annotator.rootFunction("a.b.c.d.e")
    val annotations = Annotations(Map("test" -> true))
    assert(f1(annotations) == Annotations(Map("test" -> true)))
    assert(f2(annotations) == Annotations(Map("info" -> Annotations(Map("test" -> true)))))
    assert(f3(annotations) == Annotations(Map("other" -> Annotations(
      Map("info" -> Annotations(Map("test" -> true)))))))
    assert(f4(annotations) == Annotations(Map("a" -> Annotations(
      Map("b" -> Annotations(Map("c" -> Annotations(Map("d" -> Annotations(
        Map("e" -> Annotations(Map("test" -> true)))))))))))))
  }
}
