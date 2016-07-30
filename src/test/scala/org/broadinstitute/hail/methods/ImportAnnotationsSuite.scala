package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.TestUtils._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Prop
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
      Array("table", "-i", "src/test/resources/sampleAnnotations.tsv",
        "-e", "Sample",
        "-r", "sa.`my phenotype`",
        "-t", "qPhen:Int"))

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

    assert(m("A").contains(Annotation("Newton", "C", "D", false, false)))
    assert(m("B").contains(Annotation("Turing", "C", "D", true, true)))
    assert(m("C").contains(Annotation(null, null, null, null, null)))
    assert(m("D").contains(Annotation(null, null, null, null, null)))
    assert(m("E").contains(Annotation(null, null, null, null, null)))
    assert(m("F").isEmpty)

    interceptFatal("non-numeric") {
      AnnotateSamples.run(s, Array("fam", "-i", "src/test/resources/importFamCaseControlNumericException.fam"))
    }


    s = AnnotateSamples.run(s, Array("fam", "-i", "src/test/resources/importFamQPheno.fam", "-q"))
    val m1 = qMap("sa.fam", s)

    assert(m1("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m1("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m1("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m1("D").contains(Annotation(null, null, null, null, -9.0)))
    assert(m1("E").contains(Annotation(null, null, null, null, null)))
    assert(m1("F").isEmpty)


    s = AnnotateSamples.run(s,
      Array("fam", "-i", "src/test/resources/importFamQPheno.space.m9.fam", "-q", "-d", "\\\\s+", "-m", "-9", "-r", "sa.ped"))
    val m2 = qMap("sa.ped", s)

    assert(m2("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m2("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m2("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m2("D").contains(Annotation(null, null, null, null, null)))
    assert(m2("E").contains(Annotation(null, null, null, null, 3.0)))
    assert(m2("F").isEmpty)

  }

  @Test def testSampleListAnnotator() {
    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/sample.vcf"))

    val sampleList1 = Array("foo1", "foo2", "foo3", "foo4")
    val sampleList2 = Array("C1046::HG02024", "C1046::HG02025", "C1046::HG02026",
      "C1047::HG00731", "C1047::HG00732", "C1047::HG00733", "C1048::HG02024")
    val sampleList3 = s.vds.sampleIds.toArray

    val fileRoot = tmpDir.createTempFile(prefix = "sampleListAnnotator")
    writeTable(fileRoot + "file1.txt", sc.hadoopConfiguration, sampleList1)
    writeTable(fileRoot + "file2.txt", sc.hadoopConfiguration, sampleList2)
    writeTable(fileRoot + "file3.txt", sc.hadoopConfiguration, sampleList3)

    s = AnnotateSamples.run(s, Array("list", "-i", fileRoot + "file1.txt", "-r", "sa.test1"))
    s = AnnotateSamples.run(s, Array("list", "-i", fileRoot + "file2.txt", "-r", "sa.test2"))
    s = AnnotateSamples.run(s, Array("list", "-i", fileRoot + "file3.txt", "-r", "sa.test3"))

    val (_, querier1) = s.vds.querySA("sa.test1")
    val (_, querier2) = s.vds.querySA("sa.test2")
    val (_, querier3) = s.vds.querySA("sa.test3")

    assert(s.vds.sampleIdsAndAnnotations.forall { case (sample, sa) => querier1(sa).get == false })
    assert(s.vds.sampleIdsAndAnnotations.forall { case (sample, sa) => querier3(sa).get == true })
    assert(s.vds.sampleIdsAndAnnotations.forall { case (sample, sa) => querier2(sa).get == sampleList2.contains(sample) })
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
      Array("table", "src/test/resources/variantAnnotations.tsv",
        "-t", "Rand1:Double,Rand2:Double",
        "-e", "Variant(Chromosome, Position.toInt, Ref, Alt)",
        "-c", "va.stuff = select(table, Rand1, Rand2, Gene)"))

    val q1 = anno1.vds.queryVA("va.stuff")._2
    anno1.vds.rdd
      .collect()
      .foreach {
        case (v, va, gs) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(q1(va).contains(Annotation(rand1.getOrElse(null), rand2.getOrElse(null), gene.getOrElse(null))))
      }

    val anno1alternate = AnnotateVariants.run(state,
      Array("table", "src/test/resources/variantAnnotations.alternateformat.tsv",
        "--variant-expr", "Variant(`Chromosome:Position:Ref:Alt`)",
        "-t", "Rand1:Double,Rand2:Double",
        "-c", "va.stuff = select(table, Rand1, Rand2, Gene)"))

    val anno1glob = AnnotateVariants.run(state, Array("table", "src/test/resources/variantAnnotations.split.*.tsv",
      "-t", "Rand1:Double,Rand2:Double",
      "-e", "Variant(Chromosome, Position.toInt, Ref, Alt)",
      "-c", "va.stuff = select(table, Rand1, Rand2, Gene)"))

    assert(anno1alternate.vds.same(anno1.vds))
    assert(anno1glob.vds.same(anno1.vds))
  }

  @Test def testVCFAnnotator() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = SplitMulti.run(State(sc, sqlContext, vds), noArgs)

    val anno1 = AnnotateVariants.run(state, Array("vcf", "src/test/resources/sampleInfoOnly.vcf", "--root", "va.other", "--split"))

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
            q1(va).contains(false))
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

  @Test def testImportAnnotations() {
    val s0 = State(sc, sqlContext)
    var s: State = null
    var t: State = null

    s = ImportVCF.run(s0, Array("src/test/resources/sample.vcf"))
    val sSample = SplitMulti.run(s, Array.empty[String])

    // tsv
    val importTSVFile = tmpDir.createTempFile("variantAnnotationsTSV", ".vds")
    s = ImportAnnotations.run(s0,
      Array("table", "src/test/resources/variantAnnotations.tsv",
        "-e", "Variant(Chromosome, Position, Ref, Alt)",
        "-t", "Chromosome: String",
        "--impute"))
    s = SplitMulti.run(s)
    s = AnnotateVariantsExpr.run(s, Array("-c",
      """va = {Chromosome: va.Chromosome,
        |Position: va.Position,
        |Ref: va.Ref,
        |Alt: va.Alt,
        |Rand1: va.Rand1,
        |Rand2: va.Rand2,
        |Gene: va.Gene}""".stripMargin))
    Write.run(s, Array("-o", importTSVFile))

    s = AnnotateVariants.run(sSample,
      Array("table", "src/test/resources/variantAnnotations.tsv",
        "-e", "Variant(Chromosome, Position, Ref, Alt)",
        "-t", "Chromosome: String",
        "-r", "va.stuff",
        "--impute"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importTSVFile, "-r", "va.stuff"))


    assert(s.vds.same(t.vds))

    // vcf
    val importVCFFile = tmpDir.createTempFile("variantAnnotationsVCF", ".vds")
    s = ImportVCF.run(s0, Array("src/test/resources/sampleInfoOnly.vcf"))
    s = SplitMulti.run(s)
    Write.run(s, Array("-o", importVCFFile))

    s = AnnotateVariants.run(sSample,
      Array("vcf", "src/test/resources/sampleInfoOnly.vcf",
        "-r", "va.other", "--split"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importVCFFile,
        "-r", "va.other"))

    assert(s.vds.same(t.vds))

    // json
    val importJSONFile = tmpDir.createTempFile("variantAnnotationsJSON", ".vds")

    val jsonSchema = "_0: Struct { Rand1: Double, Rand2: Double, Gene: String, contig: String, start: Int, ref: String, alt: String }"
    // FIXME better way to array-ify
    val vFields =
    """Variant(_0.contig, _0.start, _0.ref, _0.alt.split("/"))"""

    s = ImportAnnotations.run(s0,
      Array("table", "src/test/resources/importAnnot.json",
        "--variant-expr", vFields,
        "-t", jsonSchema,
        "--no-header"))
    s = AnnotateVariants.run(s, Array("expr", "-c", "va = va._0"))
    s = s.copy(vds = s.vds.copy(wasSplit = true))
    Write.run(s, Array("-o", importJSONFile))

    s = AnnotateVariants.run(sSample,
      Array("table", "src/test/resources/importAnnot.json",
        "-t", jsonSchema,
        "--variant-expr", vFields,
        "--root", "va.third",
        "--no-header"))
    s = AnnotateVariants.run(s, Array("expr", "-c", "va.third = va.third._0"))
    t = AnnotateVariants.run(sSample,
      Array("vds", "-i", importJSONFile, "-r", "va.third"))

    assert(s.vds.same(t.vds))

    val importTableFile = tmpDir.createTempFile("variantAnnotationsTable", ".tsv")

    s = ImportVCF.run(s0, Array("src/test/resources/sampleInfoOnly.vcf"))
    ExportVariants.run(s, Array(
      "-o", importTableFile,
      "-c", "v, va.info.AC, va.info.AN"
    ))
    s = AnnotateVariantsExpr.run(s, Array("-c", "va = {AC: va.info.AC, AN: va.info.AN}"))
    val checkpoint = s.vds
    s = ImportAnnotationsTable.run(s, Array(importTableFile,
      "--no-header",
      "-t", "_0: Variant, _1: Array[Int], _2: Int",
      "-c", "va.AC = table._1, va.AN = table._2",
      "-e", "_0"))

    assert(s.vds.same(checkpoint))

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

  @Test def testTables() {

    val format1 =
      """Chr Pos        Ref   Alt   Anno1   Anno2
        |22  16050036   A     C     0.123   var1
        |22  16050115   G     A     0.234   var2
        |22  16050159   C     T     0.345   var3
        |22  16050213   C     T     0.456   var4""".stripMargin

    val format2 =
      """# This line has some garbage
        |### this line too
        |## and this one
        |Chr Pos        Ref   Alt   Anno1   Anno2
        |22  16050036   A     C     0.123   var1
        |22  16050115   G     A     0.234   var2
        |22  16050159   C     T     0.345   var3
        |22  16050213   C     T     0.456   var4""".stripMargin

    val format3 =
      """# This line has some garbage
        |### this line too
        |## and this one
        |22  16050036   A     C     0.123   var1
        |22  16050115   G     A     0.234   var2
        |22  16050159   C     T     0.345   var3
        |22  16050213   C     T     0.456   var4""".stripMargin

    val format4 =
      """22,16050036,A,C,0.123,var1
        |22,16050115,G,A,0.234,var2
        |22,16050159,C,T,0.345,var3
        |22,16050213,C,T,0.456,var4""".stripMargin

    val format5 =
      """Chr Pos        Ref   Alt   OtherCol1  Anno1   OtherCol2  Anno2   OtherCol3
        |22  16050036   A     C     .          0.123   .          var1    .
        |22  16050115   G     A     .          0.234   .          var2    .
        |22  16050159   C     T     .          0.345   .          var3    .
        |22  16050213   C     T     .          0.456   .          var4    .""".stripMargin

    val format6 =
      """!asdasd
        |!!astasd
        |!!!asdagaadsa
        |22,16050036,A,C,...,0.123,...,var1,...
        |22,16050115,G,A,...,0.234,...,var2,...
        |22,16050159,C,T,...,0.345,...,var3,...
        |22,16050213,C,T,...,0.456,...,var4,...""".stripMargin

    val tmpf1 = tmpDir.createTempFile("f1", ".txt")
    val tmpf2 = tmpDir.createTempFile("f2", ".txt")
    val tmpf3 = tmpDir.createTempFile("f3", ".txt")
    val tmpf4 = tmpDir.createTempFile("f4", ".txt")
    val tmpf5 = tmpDir.createTempFile("f5", ".txt")
    val tmpf6 = tmpDir.createTempFile("f6", ".txt")

    writeTextFile(tmpf1, hadoopConf) { out => out.write(format1) }
    writeTextFile(tmpf2, hadoopConf) { out => out.write(format2) }
    writeTextFile(tmpf3, hadoopConf) { out => out.write(format3) }
    writeTextFile(tmpf4, hadoopConf) { out => out.write(format4) }
    writeTextFile(tmpf5, hadoopConf) { out => out.write(format5) }
    writeTextFile(tmpf6, hadoopConf) { out => out.write(format6) }

    val s = SplitMulti.run(State(sc, sqlContext, LoadVCF(sc, "src/test/resources/sample.vcf")))
    val fmt1 = AnnotateVariants.run(s, Array("table", tmpf1,
      "-d", "\\s+",
      "-e", "Variant(str(Chr), Pos, Ref, Alt)",
      "-c", "va = merge(va, select(table, Anno1, Anno2))",
      "--impute"))

    val fmt2 = AnnotateVariants.run(s, Array("table", tmpf2,
      "-d", "\\s+",
      "-e", "Variant(str(Chr), Pos, Ref, Alt)",
      "-c", "va = merge(va, select(table, Anno1, Anno2))",
      "--comment", "#",
      "--impute"))

    val fmt3 = AnnotateVariants.run(s, Array("table", tmpf3,
      "-d", "\\s+",
      "-e", "Variant(str(_0), _1, _2, _3)",
      "-c", "va.Anno1 = table._4, va.Anno2 = table._5",
      "--comment", "#",
      "--no-header",
      "--impute"))

    val fmt4 = AnnotateVariants.run(s, Array("table", tmpf4,
      "-d", ",",
      "-e", "Variant(str(_0), _1, _2, _3)",
      "-c", "va.Anno1 = table._4, va.Anno2 = table._5",
      "--no-header",
      "--impute"))

    val fmt5 = AnnotateVariants.run(s, Array("table", tmpf5,
      "-d", "\\s+",
      "-e", "Variant(str(Chr), Pos, Ref, Alt)",
      "-c", "va.Anno1 = table.Anno1, va.Anno2 = table.Anno2",
      "--impute"))

    val fmt6 = AnnotateVariants.run(s, Array("table", tmpf6,
      "-d", ",",
      "-e", "Variant(str(_0), _1, _2, _3)",
      "--no-header",
      "--comment", "!",
      "-c", "va.Anno1 = table._5, va.Anno2 = table._7",
      "--impute"))

    val vds1 = fmt1.vds.cache()

    assert(vds1.same(fmt2.vds))
    assert(vds1.same(fmt3.vds))
    assert(vds1.same(fmt4.vds))
    assert(vds1.same(fmt5.vds))
    assert(vds1.same(fmt6.vds))
  }

  @Test def testAnnotationsVDSReadWrite() {
    val outPath = tmpDir.createTempFile("annotationOut", ".vds")
    val p = Prop.forAll(VariantSampleMatrix.gen(sc, VSMSubgen.realistic)
      .filter(vds => vds.nVariants > 0)) { vds: VariantDataset =>

      var state = State(sc, sqlContext, vds)
      state = Write.run(state, Array("-o", outPath))

      state = AnnotateVariantsVDS.run(state, Array(
        "-i", outPath,
        "-c", "va = vds"))

      state.vds.same(vds)
    }

    p.check(count = 1)
  }
}

