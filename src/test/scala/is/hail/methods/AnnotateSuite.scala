package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils._
import is.hail.annotations._
import is.hail.check.Prop
import is.hail.expr.types._
import is.hail.io.annotators.{BedAnnotator, IntervalList}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.io.Source

class AnnotateSuite extends SparkSuite {

  @Test def testSampleTSVAnnotator() {
    val vds = hc.importVCF("src/test/resources/sample2.vcf")

    val path1 = "src/test/resources/sampleAnnotations.tsv"
    val fileMap = hadoopConf.readFile(path1) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.startsWith("Sample"))
        .map(line => {
          val split = line.split("\t")
          val f3 = split(2) match {
            case "NA" => None
            case x => Some(x.toInt)
          }
          (split(0): Annotation, (Some(split(1)), f3))
        })
        .toMap
    }

    val anno1 = vds.annotateSamplesTable(
      hc.importTable("src/test/resources/sampleAnnotations.tsv", types = Map("qPhen" -> TInt32())).keyBy("Sample"),
      root = "sa.`my phenotype`")

    val q1 = anno1.querySA("sa.`my phenotype`.Status")._2
    val q2 = anno1.querySA("sa.`my phenotype`.qPhen")._2

    assert(anno1.stringSampleIdsAndAnnotations
      .forall { case (id, sa) =>
        fileMap.get(id).forall { case (status, qphen) =>
          status.liftedZip(Option(q1(sa))).exists { case (v1, v2) => v1 == v2 } &&
            (qphen.isEmpty && Option(q2(sa)).isEmpty || qphen.liftedZip(Option(q2(sa))).exists { case (v1, v2) => v1 == v2 })
        }
      })
  }

  @Test def testSampleFamAnnotator() {
    def assertNumeric(s: String) = assert(PlinkLoader.numericRegex.findFirstIn(s).isDefined)

    def assertNonNumeric(s: String) = assert(PlinkLoader.numericRegex.findFirstIn(s).isEmpty)

    List("0", "0.0", ".0", "-01", "1e5", "1e10", "1.1e10", ".1E-10").foreach(assertNumeric)
    List("", "a", "1.", ".1.", "1e", "e", "E0", "1e1.", "1e.1", "1e1.1").foreach(assertNonNumeric)

    def qMap(query: String, vds: MatrixTable): Map[Annotation, Option[Any]] = {
      val q = vds.querySA(query)._2
      vds.sampleIds
        .zip(vds.sampleAnnotations)
        .map { case (id, sa) => (id, Option(q(sa))) }
        .toMap
    }

    var vds = hc.importVCF("src/test/resources/importFam.vcf")

    vds = vds.annotateSamplesTable(
      Table.importFam(hc, "src/test/resources/importFamCaseControl.fam"), expr = "sa.fam=table")
    val m = qMap("sa.fam", vds)

    assert(m("A").contains(Annotation("Newton", "C", "D", false, false)))
    assert(m("B").contains(Annotation("Turing", "C", "D", true, true)))
    assert(m("C").contains(Annotation(null, null, null, null, null)))
    assert(m("D").contains(Annotation(null, null, null, null, null)))
    assert(m("E").contains(Annotation(null, null, null, null, null)))
    assert(m("F").isEmpty)

    interceptFatal("non-numeric") {
      Table.importFam(hc, "src/test/resources/importFamCaseControlNumericException.fam")
    }

    vds = vds.annotateSamplesTable(
      Table.importFam(hc, "src/test/resources/importFamQPheno.fam", isQuantPheno = true), expr = "sa.fam=table")
    val m1 = qMap("sa.fam", vds)

    assert(m1("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m1("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m1("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m1("D").contains(Annotation(null, null, null, null, -9.0)))
    assert(m1("E").contains(Annotation(null, null, null, null, null)))
    assert(m1("F").isEmpty)


    vds = vds.annotateSamplesTable(
      Table.importFam(hc, "src/test/resources/importFamQPheno.space.m9.fam", isQuantPheno = true,
        delimiter = "\\\\s+", missingValue = "-9"), expr = "sa.ped = table")

    val m2 = qMap("sa.ped", vds)

    assert(m2("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m2("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m2("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m2("D").contains(Annotation(null, null, null, null, null)))
    assert(m2("E").contains(Annotation(null, null, null, null, 3.0)))
    assert(m2("F").isEmpty)

  }

  @Test def testVariantTSVAnnotator() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))

    val fileMap = hadoopConf.readFile("src/test/resources/variantAnnotations.tsv") { reader =>
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
    val kt1 = hc.importTable("src/test/resources/variantAnnotations.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("Variant = Variant(Chromosome, Position.toInt32(), Ref, Alt)")
      .keyBy("Variant")

    val anno1 = vds.annotateVariantsTable(kt1,
      expr = "va.stuff = select(table, Rand1, Rand2, Gene)")

    val q1 = anno1.queryVA("va.stuff")._2
    anno1.typedRDD[Variant]
      .collect()
      .foreach {
        case (v, (va, gs)) =>
          val (rand1, rand2, gene) = fileMap(v)
          assert(q1(va) == Annotation(rand1.getOrElse(null), rand2.getOrElse(null), gene.getOrElse(null)))
      }

    val kt2 = hc.importTable("src/test/resources/variantAnnotations.alternateformat.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("v = Variant(`Chromosome:Position:Ref:Alt`)")
      .keyBy("v")
    val anno1alternate = vds.annotateVariantsTable(kt2,
      expr = "va.stuff = select(table, Rand1, Rand2, Gene)")

    val kt3 = hc.importTable("src/test/resources/variantAnnotations.split.*.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("v = Variant(Chromosome, Position.toInt32(), Ref, Alt)")
      .keyBy("v")
    val anno1glob = vds.annotateVariantsTable(kt3,
      expr = "va.stuff = select(table, Rand1, Rand2, Gene)")

    assert(anno1alternate.same(anno1))
    assert(anno1glob.same(anno1))
  }

  @Test def testVCFAnnotator() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))

    val anno1 = vds.annotateVariantsVDS(
      // sampleInfoOnly.vcf has empty genotype schema
      SplitMulti(hc.importVCF("src/test/resources/sampleInfoOnly.vcf"),
        "va.aIndex = aIndex, va.wasSplit = wasSplit", ""),
      root = Some("va.other"))

    val otherMap = SplitMulti(hc.importVCF("src/test/resources/sampleInfoOnly.vcf"),
      // sampleInfoOnly.vcf has empty genotype schema
      "va.aIndex = aIndex, va.wasSplit = wasSplit", "")
      .variantsAndAnnotations
      .collect()
      .toMap

    val initialMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    val (_, q) = anno1.queryVA("va.other")

    anno1.rdd.collect()
      .foreach {
        case (v, (va, gs)) =>
          assert(q(va) == otherMap.getOrElse(v, null))
      }
  }

  @Test def testBedIntervalAnnotator() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
      .cache()

    val bed1r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example1.bed"), root = "va.test")
    val bed2r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example2.bed"), root = "va.test")
    val bed3r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example3.bed"), root = "va.test")

    val q1 = bed1r.queryVA("va.test")._2
    val q2 = bed2r.queryVA("va.test")._2
    val q3 = bed3r.queryVA("va.test")._2

    bed1r.variantsAndAnnotations
      .collect()
      .foreach { case (v1, va) =>
        val v = v1.asInstanceOf[Variant]
        assert(v.start <= 14000000 ||
          v.start >= 17000000 ||
          q1(va) == false)
      }

    bed2r.variantsAndAnnotations
      .collect()
      .foreach { case (v1, va) =>
        val v = v1.asInstanceOf[Variant]
        if (v.start <= 14000000)
          assert(q2(va) == "gene1")
        else if (v.start >= 17000000)
          assert(q2(va) == "gene2")
        else
          assert(q2(va) == null)
      }

    bed3r.variantsAndAnnotations
      .collect()
      .foreach { case (v1, va) =>
        val v = v1.asInstanceOf[Variant]
        if (v.start <= 14000000)
          assert(q3(va) == "gene1", v)
        else if (v.start >= 17000000)
          assert(q3(va) == "gene2", v)
        else
          assert(q3(va) == null, v)
      }

    assert(bed3r.same(bed2r))

    val int1r = vds.annotateVariantsTable(IntervalList.read(hc,
      "src/test/resources/exampleAnnotation1.interval_list"),
      root = "va.test")
    val int2r = vds.annotateVariantsTable(
      IntervalList.read(hc, "src/test/resources/exampleAnnotation2.interval_list"),
      root = "va.test")

    assert(int1r.same(bed1r))
    assert(int2r.same(bed2r))
  }

  //FIXME this test is impossible to follow
  @Test def testImportAnnotations() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")

    val sSample = SplitMulti(vds)

    // tsv
    val importTSVFile = tmpDir.createTempFile("variantAnnotationsTSV", "vds")
    vds = MatrixTable.fromTable(hc.importTable("src/test/resources/variantAnnotations.tsv",
      impute = true, types = Map("Chromosome" -> TString()))
      .annotate("v = Variant(Chromosome, Position, Ref, Alt)")
      .keyBy("v"))
    vds = SplitMulti(vds)
    vds = vds
      .annotateVariantsExpr(
        """va = {Chromosome: va.Chromosome,
          |Position: va.Position,
          |Ref: va.Ref,
          |Alt: va.Alt,
          |Rand1: va.Rand1,
          |Rand2: va.Rand2,
          |Gene: va.Gene}""".stripMargin)
    vds.write(importTSVFile)

    val kt = hc.importTable("src/test/resources/variantAnnotations.tsv",
      impute = true, types = Map("Chromosome" -> TString()))
      .annotate("v = Variant(Chromosome, Position, Ref, Alt)")
      .keyBy("v")

    vds = sSample
      .annotateVariantsTable(kt, expr = "va.stuff = table")

    var t = sSample.annotateVariantsVDS(hc.readVDS(importTSVFile), root = Some("va.stuff"))

    assert(vds.same(t))

    // json
    val importJSONFile = tmpDir.createTempFile("variantAnnotationsJSON", ".vds")
    // FIXME better way to array-ify

    val kt2 = hc.importTable("src/test/resources/importAnnot.json",
      types = Map("f0" -> TStruct("Rand1" -> TFloat64(), "Rand2" -> TFloat64(),
        "Gene" -> TString(), "contig" -> TString(), "start" -> TInt32(), "ref" -> TString(), "alt" -> TString())),
      noHeader = true)
      .annotate("""v = Variant(f0.contig, f0.start, f0.ref, f0.alt.split("/"))""")
      .keyBy("v")

    vds = MatrixTable.fromTable(kt2)
      .annotateVariantsExpr("va = va.f0")
      .filterVariantsExpr("v.isBiallelic")
    vds.write(importJSONFile)

    vds = sSample.annotateVariantsTable(kt2, root = "va.third")

    t = sSample.annotateVariantsVDS(hc.readVDS(importJSONFile), root = Some("va.third"))

    assert(vds.same(t))
  }

  @Test def testAnnotateSamples() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))

    val annoMap = vds.sampleIds.map(id => (id, 5))
      .toMap
    val vds2 = vds.filterSamples({ case (s, sa) => scala.util.Random.nextFloat > 0.5 })
      .annotateSamples(annoMap, TInt32(), "sa.test")

    val q = vds2.querySA("sa.test")._2

    vds2.sampleIds
      .zipWithIndex
      .foreach {
        case (s, i) =>
          assert(q(vds2.sampleAnnotations(i)) == 5)
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

    hadoopConf.writeTextFile(tmpf1) { out => out.write(format1) }
    hadoopConf.writeTextFile(tmpf2) { out => out.write(format2) }
    hadoopConf.writeTextFile(tmpf3) { out => out.write(format3) }
    hadoopConf.writeTextFile(tmpf4) { out => out.write(format4) }
    hadoopConf.writeTextFile(tmpf5) { out => out.write(format5) }
    hadoopConf.writeTextFile(tmpf6) { out => out.write(format6) }

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val kt1 = hc.importTable(tmpf1, separator = "\\s+", impute = true)
      .annotate("v = Variant(str(Chr), Pos, Ref, Alt)")
      .keyBy("v")
    val fmt1 = vds.annotateVariantsTable(kt1, expr = "va = merge(va, select(table, Anno1, Anno2))")

    val fmt2 = vds.annotateVariantsTable(hc.importTable(tmpf2, separator = "\\s+", impute = true,
      commentChar = Some("#"))
      .annotate("v = Variant(str(Chr), Pos, Ref, Alt)")
      .keyBy("v"),
      expr = "va = merge(va, select(table, Anno1, Anno2))")

    val fmt3 = vds.annotateVariantsTable(hc.importTable(tmpf3,
      commentChar = Some("#"), separator = "\\s+", noHeader = true, impute = true)
      .annotate("v = Variant(str(f0), f1, f2, f3)")
      .keyBy("v"),
      expr = "va.Anno1 = table.f4, va.Anno2 = table.f5")

    val fmt4 = vds.annotateVariantsTable(hc.importTable(tmpf4, separator = ",", noHeader = true, impute = true)
      .annotate("v = Variant(str(f0), f1, f2, f3)")
      .keyBy("v"),
      expr = "va.Anno1 = table.f4, va.Anno2 = table.f5")

    val fmt5 = vds.annotateVariantsTable(hc.importTable(tmpf5, separator = "\\s+", impute = true, missing = ".")
      .annotate("v = Variant(str(Chr), Pos, Ref, Alt)")
      .keyBy("v"),
      expr = "va.Anno1 = table.Anno1, va.Anno2 = table.Anno2")

    val fmt6 = vds.annotateVariantsTable(hc.importTable(tmpf6,
      noHeader = true, impute = true, separator = ",", commentChar = Some("!"))
      .annotate("v = Variant(str(f0), f1, f2, f3)")
      .keyBy("v"),
      expr = "va.Anno1 = table.f5, va.Anno2 = table.f7")

    val vds1 = fmt1.cache()

    assert(vds1.same(fmt2))
    assert(vds1.same(fmt3))
    assert(vds1.same(fmt4))
    assert(vds1.same(fmt5))
    assert(vds1.same(fmt6))
  }

  @Test def testAnnotationsVDSReadWrite() {
    val outPath = tmpDir.createTempFile("annotationOut", ".vds")
    val p = Prop.forAll(MatrixTable.gen(hc, VSMSubgen.realistic)
      .filter(vds => vds.countVariants > 0)) { vds: MatrixTable =>

      vds.annotateVariantsVDS(vds, code = Some("va = vds")).same(vds)
    }

    p.check()
  }

  @Test def testPositions() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample2.vcf"))

    val kt = hc.importTable("src/test/resources/sample2_va_positions.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("loc = Locus(Chromosome, Position.toInt32())")
      .keyBy("loc")

    val byPosition = vds.annotateVariantsTable(kt, expr = "va.stuff = select(table, Rand1, Rand2)")

    val kt2 = hc.importTable("src/test/resources/sample2_va_nomulti.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("v = Variant(Chromosome, Position.toInt32(), Ref, Alt)")
      .keyBy("v")
    val byVariant = vds.annotateVariantsTable(kt2,
      expr = "va.stuff = select(table, Rand1, Rand2)")

    assert(byPosition.same(byVariant))
  }
}

