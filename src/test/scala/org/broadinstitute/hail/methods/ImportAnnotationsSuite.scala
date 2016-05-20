package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.TestUtils._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr.TInt
import org.broadinstitute.hail.io.annotators.SampleFamAnnotator
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

import scala.io.Source

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
      Array("table", "-i", "src/test/resources/sampleAnnotations.tsv", "-s", "Sample", "-r", "sa.`my phenotype`", "-t", "qPhen:Int"))

    val q1 = anno1.vds.querySA("sa.`my phenotype`.Status")._2
    val q2 = anno1.vds.querySA("sa.`my phenotype`.qPhen")._2

    anno1.vds.metadata.sampleIds.zip(anno1.vds.metadata.sampleAnnotations)
      .forall {
        case (id, sa) =>
          !fileMap.contains(id) ||
            (q1(sa).contains(fileMap(id)._1) && q2(sa).contains(fileMap(id)._2))
      }
  }

  @Test def testSampleFamAnnotator() {
    def assertNumeric(s: String) = assert(SampleFamAnnotator.numericRegex.findFirstIn(s).isDefined)
    def assertNonNumeric(s: String) = assert(SampleFamAnnotator.numericRegex.findFirstIn(s).isEmpty)

    List("0", "0.0", ".0", "-01", "1e5", "1e10", "1.1e10", ".1E-10").foreach(assertNumeric)
    List("", "a", "1.", ".1.", "1e", "e", "E0", "1e1.", "1e.1", "1e1.1").foreach(assertNonNumeric)

    def qMap(query: String, s: State): Map[String, Option[Any]] = {
      val q = s.vds.querySA(query)._2
      s.vds.sampleIds
        .zip(s.vds.sampleAnnotations)
        .map { case (id, sa) => (id, q(sa)) }
        .toMap
    }

    val vds = LoadVCF(sc, "src/test/resources/importFam.vcf")
    var s = State(sc, sqlContext, vds)


    s = AnnotateSamples.run(s, Array("fam", "-i", "src/test/resources/importFamCaseControl.fam"))
    val m = qMap("sa.fam", s)

    assert(m("A").contains(Annotation("Newton", "C", "D", true, false)))
    assert(m("B").contains(Annotation("Turing", "C", "D", false, true)))
    assert(m("C").contains(Annotation(null, null, null, null, null)))
    assert(m("D").contains(Annotation(null, null, null, null, null)))
    assert(m("E").contains(Annotation(null, null, null, null, null)))
    assert(m("F").isEmpty)

    interceptFatal("non-numeric") {
      AnnotateSamples.run(s, Array("fam", "-i", "src/test/resources/importFamCaseControlNumericException.fam"))
    }


    s = AnnotateSamples.run(s, Array("fam", "-i", "src/test/resources/importFamQPheno.fam", "-q"))
    val m1 = qMap("sa.fam", s)

    assert(m1("A").contains(Annotation("Newton", "C", "D", true, 1.0)))
    assert(m1("B").contains(Annotation("Turing", "C", "D", false, 2.0)))
    assert(m1("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m1("D").contains(Annotation(null, null, null, null, -9.0)))
    assert(m1("E").contains(Annotation(null, null, null, null, null)))
    assert(m1("F").isEmpty)


    s = AnnotateSamples.run(s,
      Array("fam", "-i", "src/test/resources/importFamQPheno.space.m9.fam", "-q", "-d", "\\\\s+", "-m", "-9", "-r", "sa.ped"))
    val m2 = qMap("sa.ped", s)

    assert(m2("A").contains(Annotation("Newton", "C", "D", true, 1.0)))
    assert(m2("B").contains(Annotation("Turing", "C", "D", false, 2.0)))
    assert(m2("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m2("D").contains(Annotation(null, null, null, null, null)))
    assert(m2("E").contains(Annotation(null, null, null, null, 3.0)))
    assert(m2("F").isEmpty)

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
      Array("table", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    val q1 = anno1.vds.queryVA("va.stuff")._2
    anno1.vds.rdd
      .collect()
      .foreach {
        case (v, va, gs) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(q1(va).contains(Annotation(rand1.getOrElse(null), rand2.getOrElse(null), gene.getOrElse(null))))
      }

    val anno1alternate = AnnotateVariants.run(state,
      Array("table", "src/test/resources/variantAnnotations.alternateformat.tsv", "--vcolumns",
        "Chromosome:Position:Ref:Alt", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    val anno1glob = AnnotateVariants.run(state, Array("table", "src/test/resources/variantAnnotations.split.*.tsv",
      "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))

    assert(anno1alternate.vds.same(anno1.vds))
    assert(anno1glob.vds.same(anno1.vds))
  }

  @Test def testVCFAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val anno1 = AnnotateVariants.run(state, Array("vcf", "src/test/resources/sampleInfoOnly.vcf", "--root", "va.other"))

    val otherMap = SplitMulti.run(State(sc, sqlContext, LoadVCF(sc, "src/test/resources/sampleInfoOnly.vcf")), Array[String]())
      .vds
      .variantsAndAnnotations
      .collect()
      .toMap

    val initialMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    val q = anno1.vds.queryVA("va.other")._2

    anno1.vds.rdd.collect()
      .foreach {
        case (v, va, gs) =>
          assert(q(va) == otherMap.get(v))
      }
  }

  @Test def testBedIntervalAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = Cache.run(SplitMulti.run(State(sc, sqlContext, vds), noArgs), noArgs)

    val bed1r = AnnotateVariants.run(state, Array("bed", "-i", "src/test/resources/example1.bed", "-r", "va.test"))
    val bed2r = AnnotateVariants.run(state, Array("bed", "-i", "src/test/resources/example2.bed", "-r", "va.test"))
    val bed3r = AnnotateVariants.run(state, Array("bed", "-i", "src/test/resources/example3.bed", "-r", "va.test"))

    val q1 = bed1r.vds.queryVA("va.test")._2
    val q2 = bed2r.vds.queryVA("va.test")._2
    val q3 = bed3r.vds.queryVA("va.test")._2

    bed1r.vds.variantsAndAnnotations
      .collect()
      .foreach {
        case (v, va) =>
          assert(v.start <= 14000000 ||
            v.start >= 17000000 ||
            q1(va).isEmpty)
      }

    bed2r.vds.variantsAndAnnotations
      .collect()
      .foreach {
        case (v, va) =>
          (v.start <= 14000000 && q2(va).contains("gene1")) ||
            (v.start >= 17000000 && q2(va).contains("gene2")) ||
            q2(va).isEmpty
      }

    assert(bed3r.vds.same(bed2r.vds))

    val int1r = AnnotateVariants.run(state, Array("intervals", "-i", "src/test/resources/exampleAnnotation1.interval_list",
      "-r", "va.test"))
    val int2r = AnnotateVariants.run(state, Array("intervals", "-i", "src/test/resources/exampleAnnotation2.interval_list",
      "-r", "va.test"))

    assert(int1r.vds.same(bed1r.vds))
    assert(int2r.vds.same(bed2r.vds))
  }

  @Test def testSerializedAnnotator() {
    val s0 = State(sc, sqlContext)
    var s: State = null
    var t: State = null

    s = ImportVCF.run(s0, Array("src/test/resources/sample.vcf"))
    val sSample = SplitMulti.run(s, Array.empty[String])

    // tsv
    val importTSVFile = tmpDir.createTempFile("variantAnnotationsTSV", ".vds")
    s = ImportAnnotations.run(s0,
      Array("table", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double"))
    Write.run(s, Array("-o", importTSVFile))

    s = AnnotateVariants.run(sSample,
      Array("table", "src/test/resources/variantAnnotations.tsv", "-t", "Rand1:Double,Rand2:Double", "-r", "va.stuff"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importTSVFile, "-r", "va.stuff"))
    assert(s.vds.same(t.vds))

    // vcf
    val importVCFFile = tmpDir.createTempFile("variantAnnotationsVCF", ".vds")
    s = ImportVCF.run(s0, Array("src/test/resources/sampleInfoOnly.vcf"))
    s = SplitMulti.run(s)
    Write.run(s, Array("-o", importVCFFile))

    s = AnnotateVariants.run(sSample,
      Array("vcf", "src/test/resources/sampleInfoOnly.vcf", "--root", "va.other"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importVCFFile, "-r", "va.other"))

    assert(s.vds.same(t.vds))

    // json
    val importJSONFile = tmpDir.createTempFile("variantAnnotationsJSON", ".vds")

    val jsonSchema = "Struct { Rand1: Double, Rand2: Double, Gene: String, contig: String, start: Int, ref: String, alt: String }"
    // FIXME better way to array-ify
    val vFields = """root.contig, root.start, root.ref, root.alt.split("/")"""

    s = ImportAnnotations.run(s0,
      Array("json", "src/test/resources/importAnnot.json", "--vfields", vFields, "-t", jsonSchema))
    Write.run(s, Array("-o", importJSONFile))

    s = AnnotateVariants.run(sSample,
      Array("json", "src/test/resources/importAnnot.json", "-t", jsonSchema, "--vfields", vFields, "--root", "va.third"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importJSONFile, "-r", "va.third"))

    assert(s.vds.same(t.vds))
  }

  @Test def testAnnotateSamples() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val annoMap = vds.sampleIds.map(id => (id, 5))
      .toMap
    val vds2 = vds.filterSamples({ case (s, sa) => scala.util.Random.nextFloat > 0.5 })
      .annotateSamples(annoMap, TInt, List("test"))

    val q = vds2.querySA("sa.test")._2

    vds2.sampleIds
      .zipWithIndex
      .foreach {
        case (s, i) =>
          assert(q(vds2.sampleAnnotations(i)) == Some(5))
      }

  }
}

