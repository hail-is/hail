package is.hail.io

import is.hail.check.Prop._
import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.io.vcf.{ExportVCF, LoadVCF}
import is.hail.variant._
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row

class ImportVCFSuite extends SparkSuite {

  @Test def testInfo() {
    assert(hc.importVCF("src/test/resources/infochar.vcf").countVariants() == 1)
  }

  @Test def lineRef() {
    val line1 = "20\t10280082\t.\tA\tG\t844.69\tPASS\tAC=1;..."
    assert(LoadVCF.lineRef(line1) == "A")

    val line2 = "20\t13561632\t.\tTAA\tT\t89057.4\tPASS\tAC=2;..."
    assert(LoadVCF.lineRef(line2) == "TAA")

    assert(LoadVCF.lineRef("") == "")

    assert(LoadVCF.lineRef("this\tis\ta") == "")

    assert(LoadVCF.lineRef("20\t0\t.\t") == "")

    assert(LoadVCF.lineRef("20\t0\t.\t\t") == "")

    assert(LoadVCF.lineRef("\t\t\tabcd") == "abcd")
  }

  @Test def testGlob() {
    val n1 = hc.importVCF("src/test/resources/sample.vcf").countVariants()
    val n2 = hc.importVCF("src/test/resources/samplepart*.vcf").countVariants()
    assert(n1 == n2)
  }

  @Test def testUndeclaredInfo() {
    val vds = hc.importVCF("src/test/resources/undeclaredinfo.vcf")

    assert(vds.rowType.getOption("info").isDefined)
    assert(vds.rowType.getOption("info", "undeclared").isEmpty)
    assert(vds.rowType.getOption("info", "undeclaredFlag").isEmpty)
    val infoQuerier = vds.rowType.query("info")

    val anno = vds
      .rdd
      .map { case (v, (va, gs)) => va }
      .collect()
      .head

    assert(infoQuerier(anno) != null)
  }

  @Test def testMalformed() {
    // FIXME abstract
    val e = intercept[SparkException] {
      hc.importVCF("src/test/resources/malformed.vcf").countVariants()
    }
    assert(e.getMessage.contains("invalid character"))
  }
  
  @Test def testHaploid() {
    val vds = hc.importVCF("src/test/resources/haploid.vcf")
    val r = vds
      .expand()
      .collect()
      .map { case (v, s, g) => ((v, s), g) }
      .toMap

    val v1 = Row(Locus("X", 16050036), IndexedSeq("A", "C"))
    val v2 = Row(Locus("X", 16061250), IndexedSeq("T", "A", "C"))

    val s1 = "C1046::HG02024"
    val s2 = "C1046::HG02025"

    assert(r(v1, s1) == Genotype(
      Call2(0, 0),
      Array(10, 0),
      10,
      44,
      Array(0, 44, 180)
    ))
    assert(r(v1, s2) == Genotype(
      Call1(1),
      Array(0, 6),
      7,
      70,
      Array(70, 0)
    ))
    assert(r(v2, s1) == Genotype(
      Call2(2, 2),
      Array(0, 0, 11),
      11,
      33,
      Array(396, 402, 411, 33, 33, 0)
    ))
    assert(r(v2, s2) == Genotype(
      Call1(2),
      Array(0, 0, 9),
      9,
      24,
      Array(24, 40, 0)
    ))
  }

  @Test def testGeneric() {
    val path = tmpDir.createTempFile(extension = ".vds")
    val vcf = "src/test/resources/sample.vcf.bgz"
    val gds = hc.importVCF(vcf)

    gds.write(path)

    val path2 = tmpDir.createTempFile(extension = ".vds")
    val vcf2 = "src/test/resources/generic.vcf"
    val gds2 = hc.importVCF(vcf2, callFields = Set("GT", "GTA", "GTZ"))
    gds2.write(path2)

    val v1 = Row(Locus("X", 16050036), IndexedSeq("A", "C"))
    val v2 = Row(Locus("X", 16061250), IndexedSeq("T", "A", "C"))

    val s1 = "C1046::HG02024"
    val s2 = "C1046::HG02025"

    val (_, querierGT) = gds2.queryGA("g.GT")
    val (_, querierGTA) = gds2.queryGA("g.GTA")
    val (_, querierGTZ) = gds2.queryGA("g.GTZ")

    val r2 = gds2
      .expand()
      .collect()
      .map { case (v, s, g) =>
        ((v, s), (querierGT(g), querierGTA(g), querierGTZ(g)))
      }.toMap

    assert(r2(v1, s1) == (Call2(0, 0), null, Call2(0, 1)))
    assert(r2(v1, s2) == (Call1(1), null, Call1(0)))
    assert(r2(v2, s1) == (Call2(2, 2), Call2(2, 1), Call2(1, 1)))
    assert(r2(v2, s2) == (Call1(2), null, Call1(1)))

    import is.hail.io.vcf.HtsjdkRecordReader._
    assert(parseCall("0/0", 2) == Call2(0, 0))
    assert(parseCall("1/0", 2) == Call2(1, 0))
    assert(parseCall("0", 2) == Call1(0))
    assert(parseCall(".", 2) == null)
    assert(parseCall("./.", 2) == null)
    assert(parseCall("1|0", 2) == Call2(1, 0, phased = true))
    assert(parseCall("0|1", 2) == Call2(0, 1, phased = true))
    intercept[HailException] {
      parseCall("./0", 2) == Call2(0, 0)
    }
    intercept[HailException] {
      parseCall("""0\0""", 2) == Call2(0, 0)
    }
  }

  @Test def testMissingInfo() {
    val vds = hc.importVCF("src/test/resources/missingInfoArray.vcf")

    val variants = vds.queryVariants("variants.map(_ => va.locus).collect()")._1.asInstanceOf[IndexedSeq[Locus]]
    val foo = vds.queryVariants("variants.map(v => va.info.FOO).collect()")._1.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]
    val bar = vds.queryVariants("variants.map(v => va.info.BAR).collect()")._1.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Double]]]

    val vMap = (variants, foo, bar).zipped.map { case (v, f, b) => (v, (f, b)) }.toMap

    assert(vMap == Map(
      Locus("X", 16050036) -> (IndexedSeq(1, null), IndexedSeq(2, null, null)),
      Locus("X", 16061250) -> (IndexedSeq(null, 2, null), IndexedSeq(null, 1.0, null))
    ))
  }

  @Test def randomExportImportIsIdentity() {
    forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>

      val truth = {
        val f = tmpDir.createTempFile(extension="vcf")
        ExportVCF(vds, f)
        hc.importVCF(f, gr = vds.genomeReference)
      }

      val actual = {
        val f = tmpDir.createTempFile(extension="vcf")
        ExportVCF(truth, f)
        hc.importVCF(f, gr = vds.genomeReference)
      }

      truth.same(actual)
    }.check()
  }

  @Test def notIdenticalHeaders() {
    val tmp1 = tmpDir.createTempFile("sample1", ".vcf")
    val tmp2 = tmpDir.createTempFile("sample2", ".vcf")
    val tmp3 = tmpDir.createTempFile("sample3", ".vcf")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val sampleIds = vds.stringSampleIds
    ExportVCF(vds.filterSamplesList(Set(Annotation(sampleIds(0)))), tmp1)
    ExportVCF(vds.filterSamplesList(Set(Annotation(sampleIds(1)))), tmp2)
    assert(intercept[SparkException] (hc.importVCFs(Array(tmp1, tmp2))).getMessage.contains("invalid sample ids"))

    ExportVCF(vds.filterSamplesList(Set(sampleIds(0),sampleIds(1)).map(Annotation(_))), tmp3)
    assert(intercept[SparkException] (hc.importVCFs(Array(tmp1, tmp3))).getMessage.contains("invalid sample ids"))

    // no error thrown if user provides own header
    hc.importVCFs(Array(tmp1, tmp2), headerFile = Some("src/test/resources/sample.vcf"))
  }
}
