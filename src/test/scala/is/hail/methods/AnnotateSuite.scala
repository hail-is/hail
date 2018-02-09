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
      root = "my phenotype")

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
      vds.stringSampleIds
        .zip(vds.colValues)
        .map { case (id, sa) => (id, Option(q(sa))) }
        .toMap
    }

    var vds = hc.importVCF("src/test/resources/importFam.vcf")

    vds = vds.annotateSamplesTable(
      Table.importFam(hc, "src/test/resources/importFamCaseControl.fam"), root = "fam")
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
      Table.importFam(hc, "src/test/resources/importFamQPheno.fam", isQuantPheno = true), root = "fam")
    val m1 = qMap("sa.fam", vds)

    assert(m1("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m1("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m1("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m1("D").contains(Annotation(null, null, null, null, -9.0)))
    assert(m1("E").contains(Annotation(null, null, null, null, null)))
    assert(m1("F").isEmpty)


    vds = vds.annotateSamplesTable(
      Table.importFam(hc, "src/test/resources/importFamQPheno.space.m9.fam", isQuantPheno = true,
        delimiter = "\\\\s+", missingValue = "-9"), root = "ped")

    val m2 = qMap("sa.ped", vds)

    assert(m2("A").contains(Annotation("Newton", "C", "D", false, 1.0)))
    assert(m2("B").contains(Annotation("Turing", "C", "D", true, 2.0)))
    assert(m2("C").contains(Annotation(null, null, null, null, 0.0)))
    assert(m2("D").contains(Annotation(null, null, null, null, null)))
    assert(m2("E").contains(Annotation(null, null, null, null, 3.0)))
    assert(m2("F").isEmpty)

  }

  @Test def testBedIntervalAnnotator() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
      .cache()

    val bed1r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example1.bed"), root = "test").annotateVariantsExpr("test = isDefined(va.test)")
    val bed2r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example2.bed"), root = "test").annotateVariantsExpr("test = va.test.target")
    val bed3r = vds.annotateVariantsTable(BedAnnotator(hc, "src/test/resources/example3.bed"), root = "test").annotateVariantsExpr("test = va.test.target")

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
      "src/test/resources/exampleAnnotation1.interval_list"), "test")
        .annotateVariantsExpr("test = isDefined(va.test)")
    val int2r = vds.annotateVariantsTable(
      IntervalList.read(hc, "src/test/resources/exampleAnnotation2.interval_list"), "test")
        .annotateVariantsExpr("test = va.test.target")

    assert(int1r.same(bed1r))
    assert(int2r.same(bed2r))
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
      .annotate("locus = Locus(str(Chr), Pos), alleles = [Ref, Alt]")
      .keyBy("locus", "alleles")
    val fmt1 = vds.annotateVariantsTable(kt1, "table").annotateVariantsExpr("table = {Anno1: va.table.Anno1, Anno2: va.table.Anno2}")

    val fmt2 = vds.annotateVariantsTable(hc.importTable(tmpf2, separator = "\\s+", impute = true,
      commentChar = Some("#"))
      .annotate("locus = Locus(str(Chr), Pos), alleles = [Ref, Alt]")
      .keyBy("locus", "alleles"),
      "table").annotateVariantsExpr("table = {Anno1: va.table.Anno1, Anno2: va.table.Anno2}")

    val fmt3 = vds.annotateVariantsTable(hc.importTable(tmpf3,
      commentChar = Some("#"), separator = "\\s+", noHeader = true, impute = true)
      .annotate("locus = Locus(str(f0), f1), alleles = [f2, f3]")
      .keyBy("locus", "alleles"),
      "table").annotateVariantsExpr("table = {Anno1: va.table.f4, Anno2: va.table.f5}")

    val fmt4 = vds.annotateVariantsTable(hc.importTable(tmpf4, separator = ",", noHeader = true, impute = true)
      .annotate("locus = Locus(str(f0), f1), alleles = [f2, f3]")
      .keyBy("locus", "alleles"),
      "table").annotateVariantsExpr("table = {Anno1: va.table.f4, Anno2: va.table.f5}")

    val fmt5 = vds.annotateVariantsTable(hc.importTable(tmpf5, separator = "\\s+", impute = true, missing = ".")
      .annotate("locus = Locus(str(Chr), Pos), alleles = [Ref, Alt]")
      .keyBy("locus", "alleles"),
      "table").annotateVariantsExpr("table = {Anno1: va.table.Anno1, Anno2: va.table.Anno2}")

    val fmt6 = vds.annotateVariantsTable(hc.importTable(tmpf6,
      noHeader = true, impute = true, separator = ",", commentChar = Some("!"))
      .annotate("locus = Locus(str(f0), f1), alleles = [f2, f3]")
      .keyBy("locus", "alleles"),
      "table").annotateVariantsExpr("table = {Anno1: va.table.f5, Anno2: va.table.f7}")

    val vds1 = fmt1.cache()

    assert(vds1.same(fmt2))
    assert(vds1.same(fmt3))
    assert(vds1.same(fmt4))
    assert(vds1.same(fmt5))
    assert(vds1.same(fmt6))
  }

  @Test def testPositions() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample2.vcf"))

    val kt = hc.importTable("src/test/resources/sample2_va_positions.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("locus = Locus(Chromosome, Position.toInt32())")
      .keyBy("locus")

    val byPosition = vds.annotateVariantsTable(kt, "stuff").annotateVariantsExpr("stuff = {Rand1: va.stuff.Rand1, Rand2: va.stuff.Rand2}")

    val kt2 = hc.importTable("src/test/resources/sample2_va_nomulti.tsv",
      types = Map("Rand1" -> TFloat64(), "Rand2" -> TFloat64()))
      .annotate("loc = Locus(Chromosome, Position.toInt32()), alleles = [Ref, Alt]")
      .keyBy("loc", "alleles")
    val byVariant = vds.annotateVariantsTable(kt2,
      "stuff").annotateVariantsExpr("stuff = {Rand1: va.stuff.Rand1, Rand2: va.stuff.Rand2}")

    assert(byPosition.same(byVariant))
  }
}

