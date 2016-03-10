package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.io.Source
import org.broadinstitute.hail.Utils._

class ImportAnnotationsSuite extends SparkSuite {

  @Test def testSampleTSVAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")

    val state = State(sc, sqlContext, vds)

    val path1 = "src/test/resources/sampleAnnotations.tsv"
    val fileMap = readFile(path1, sc.hadoopConfiguration) { reader =>
      Source.fromInputStream(reader)
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
    }

    val anno1 = AnnotateSamples.run(state,
      Array("-c", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-r", "sa.phenotype", "-t", "qPhen:Int"))

    val q1 = vds.querySA("phenotype", "Status")
    val q2 = vds.querySA("phenotype", "qPhen")

    anno1.vds.metadata.sampleIds.zip(anno1.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) ||
            ((Some(fileMap(id)._1) == q1(sa)) && (Some(fileMap(id)._2) == q2(sa)))
      }
  }

  @Test def testVariantTSVAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val fileMap = readFile("src/test/resources/variantAnnotations.tsv", sc.hadoopConfiguration) { reader =>
      Source.fromInputStream(reader)
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
    }
    val anno1 = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    val q1 = anno1.vds.queryVA("stuff")
    anno1.vds.rdd
      .collect()
      .foreach {
        case (v, va, gs) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(q1(va) == Some(Annotation(rand1, rand2, gene)))
      }

    val anno1alternate = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.alternateformat.tsv", "--vcolumns",
        "Chromosome:Position:Ref:Alt", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    assert(anno1alternate.vds.same(anno1.vds))
  }

  @Test def testVCFAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val anno1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/sampleInfoOnly.vcf", "--root", "va.other"))

    val otherMap = LoadVCF(sc, "src/test/resources/sampleInfoOnly.vcf")
      .variantsAndAnnotations
      .collect()
      .toMap

    val initialMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    val q = vds.queryVA("other")

    anno1.vds.eraseSplit.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          assert(q(va) == otherMap.get(v))
      }
  }

  @Test def testBedIntervalAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = Cache.run(SplitMulti.run(State(sc, sqlContext, vds), noArgs), noArgs)

    val bed1r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example1.bed", "-r", "va.bed"))
    val bed2r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example2.bed", "-r", "va.bed"))
    val bed3r = AnnotateVariants.run(state, Array("-c", "src/test/resources/example3.bed", "-r", "va.bed"))

    val q1 = bed1r.vds.queryVA("bed", "BedTest")
    val q2 = bed2r.vds.queryVA("bed", "BedTest")
    val q3 = bed3r.vds.queryVA("bed", "BedTest")

    bed1r.vds.variantsAndAnnotations
      .collect()
      .foreach {
        case (v, va) =>
          assert(v.start <= 14000000 ||
            v.start >= 17000000 ||
            q1(va) == Some(true))
      }
//
//    bed2r.vds.variantsAndAnnotations
//      .collect()
//      .foreach {
//        case (v, va) =>
//          v.start <= 14000000 ||
//            v.start >= 17000000 ||
//            (va.contains("BedTest") &&
//              va.get[String]("BedTest") == map1r(v).get[Annotations]("bed").get[String]("BedTest"))
//      }
//
//    assert(bed3.vds.same(bed2.vds))
//    assert(bed3r.vds.same(bed2r.vds))
//
//    val int1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation1.interval_list",
//      "-i", "BedTest"))
//    val int1r = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation1.interval_list",
//      "-i", "BedTest", "-r", "va.bed"))
//    val int2 = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation2.interval_list",
//      "-i", "BedTest"))
//    val int2r = AnnotateVariants.run(state, Array("-c", "src/test/resources/exampleAnnotation2.interval_list",
//      "-i", "BedTest", "-r", "va.bed"))
//
//    val bedMap = bed1.vds.variantsAndAnnotations.collect().toMap
//
//    assert(int1.vds.same(bed1.vds))
//    assert(int1r.vds.same(bed1r.vds))
//    assert(int2.vds.same(bed2.vds))
//    assert(int2r.vds.same(bed2r.vds))
  }

  @Test def testSerializedAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val tsv1r = AnnotateVariants.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    val vcf1 = AnnotateVariants.run(state, Array("-c", "src/test/resources/sampleInfoOnly.vcf", "--root", "va.other"))

    ConvertAnnotations.run(state,
      Array("-c", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double",
        "-o", "/tmp/variantAnnotationsTSV.ser"))
    ConvertAnnotations.run(state,
      Array("-c", "src/test/resources/sampleInfoOnly.vcf", "-o", "/tmp/variantAnnotationsVCF.ser"))

    val tsvSer1r = AnnotateVariants.run(state,
      Array("-c", "/tmp/variantAnnotationsTSV.ser", "-r", "va.stuff"))

    val vcfSer1 = AnnotateVariants.run(state,
      Array("-c", "/tmp/variantAnnotationsVCF.ser", "-r", "va.other"))

    assert(tsv1r.vds.same(tsvSer1r.vds))
    assert(vcf1.vds.same(vcfSer1.vds))
  }

  @Test def testOverwriteBehavior() {


    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)
    //
    //    val annotator = new DummyAnnotator
    //    vds.annotateVariants(annotator)
    //      .variantsAndAnnotations
    //      .collect()
    //      .foreach { case (v, va) =>
    //        if (v.start % 2 == 0)
    //          assert(va.get[Annotations]("info").getOption[Int]("AC").isEmpty)
    //        else
    //          assert(va.get[Annotations]("info").getOption[Int]("AC") == Some(0))
  }
}

