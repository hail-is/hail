package is.hail.io

import is.hail.check.Prop._
import is.hail.SparkSuite
import is.hail.io.vcf.{HtsjdkRecordReader, LoadVCF}
import is.hail.variant.{Call, Genotype, VSMSubgen, Variant, VariantSampleMatrix}
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._

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

  @Test def symbolicOrSV() {
    val vds = hc.importVCF("src/test/resources/symbolicVariant.vcf")
    val n = vds.countVariants()

    assert(n == 1)
  }

  @Test def testGlob() {
    val n1 = hc.importVCF("src/test/resources/sample.vcf").countVariants()
    val n2 = hc.importVCF("src/test/resources/samplepart*.vcf").countVariants()
    assert(n1 == n2)
  }

  @Test def testUndeclaredInfo() {
    val vds = hc.importVCF("src/test/resources/undeclaredinfo.vcf")

    assert(vds.vaSignature.getOption("info").isDefined)
    assert(vds.vaSignature.getOption("info", "undeclared").isEmpty)
    assert(vds.vaSignature.getOption("info", "undeclaredFlag").isEmpty)
    val infoQuerier = vds.vaSignature.query("info")

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

    val v1 = Variant("X", 16050036, "A", "C")
    val v2 = Variant("X", 16061250, "T", Array("A", "C"))

    val s1 = "C1046::HG02024"
    val s2 = "C1046::HG02025"

    assert(r(v1, s1) == Genotype(
      Some(0),
      Some(Array(10, 0)),
      Some(10),
      Some(44),
      Some(Array(0, 44, 180))
    ))
    assert(r(v1, s2) == Genotype(
      Some(2),
      Some(Array(0, 6)),
      Some(7),
      Some(70),
      Some(Array(70, HtsjdkRecordReader.haploidNonsensePL, 0))
    ))
    assert(r(v2, s1) == Genotype(
      Some(5),
      Some(Array(0, 0, 11)),
      Some(11),
      Some(33),
      Some(Array(396, 402, 411, 33, 33, 0))
    ))
    assert(r(v2, s2) == Genotype(
      Some(5),
      Some(Array(0, 0, 9)),
      Some(9),
      Some(24),
      Some(Array(24, HtsjdkRecordReader.haploidNonsensePL, 40, HtsjdkRecordReader.haploidNonsensePL,
        HtsjdkRecordReader.haploidNonsensePL, 0))
    ))
  }

  @Test def testGeneric() {
    val path = tmpDir.createTempFile(extension = ".vds")
    val vcf = "src/test/resources/sample.vcf.bgz"
    val gds = hc.importVCFGeneric(vcf)
    val vds = hc.importVCF(vcf)

    val (_, qGT) = gds.queryGA("g.GT")
    val callVDS = vds.expand().map { case (v, s, g) => ((v, s), Genotype.call(g)) }
    val callGDS = gds.expand().map { case (v, s, g) => ((v.asInstanceOf[Variant], s), qGT(g).asInstanceOf[Call]) }

    assert(callVDS.fullOuterJoin(callGDS).forall { case ((v, s), (c1, c2)) => c1 == c2 })

    gds.write(path)

    val path2 = tmpDir.createTempFile(extension = ".vds")
    val vcf2 = "src/test/resources/generic.vcf"
    val gds2 = hc.importVCFGeneric(vcf2, callFields = Set("GT", "GTA", "GTZ"))
    gds2.write(path2)

    val v1 = Variant("X", 16050036, "A", "C")
    val v2 = Variant("X", 16061250, "T", Array("A", "C"))

    val s1 = "C1046::HG02024"
    val s2 = "C1046::HG02025"

    val (_, querierGT) = gds2.queryGA("g.GT")
    val (_, querierGTA) = gds2.queryGA("g.GTA")
    val (_, querierGTZ) = gds2.queryGA("g.GTZ")

    val r2 = gds2
      .expand()
      .collect()
      .map { case (v, s, g) => ((v, s), (querierGT(g), querierGTA(g), querierGTZ(g))) }
      .toMap

    assert(r2(v1, s1) == (0, null, 1))
    assert(r2(v1, s2) == (2, null, 0))
    assert(r2(v2, s1) == (5, 4, 2))
    assert(r2(v2, s2) == (5, null, 2))

    import is.hail.io.vcf.HtsjdkRecordReader._
    assert(parseCall("0/0", 2) == Call(0))
    assert(parseCall("1/0", 2) == Call(1))
    assert(parseCall("0", 2) == Call(0))
    assert(parseCall(".", 2) == null)
    assert(parseCall("./.", 2) == null)
    intercept[HailException] {
      parseCall("./0", 2) == Call(0)
    }
    intercept[HailException] {
      parseCall("""0\0""", 2) == Call(0)
    }
  }

  @Test def testMissingInfo() {
    val vds = hc.importVCF("src/test/resources/missingInfoArray.vcf")

    val variants = vds.queryVariants("variants.collect()")._1.asInstanceOf[IndexedSeq[Variant]]
    val foo = vds.queryVariants("variants.map(v => va.info.FOO).collect()")._1.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Integer]]]
    val bar = vds.queryVariants("variants.map(v => va.info.BAR).collect()")._1.asInstanceOf[IndexedSeq[IndexedSeq[java.lang.Double]]]

    val vMap = (variants, foo, bar).zipped.map { case (v, f, b) => (v, (f, b)) }.toMap

    assert(vMap == Map(
      Variant("X", 16050036, "A", "C") -> (IndexedSeq(1, null), IndexedSeq(2, null, null)),
      Variant("X", 16061250, "T", Array("A", "C")) -> (IndexedSeq(null, 2, null), IndexedSeq(null, 1.0, null))
    ))
  }

  @Test def randomExportImportIsIdentity() {
    forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vds =>

      val truth = {
        val f = tmpDir.createTempFile(extension="vcf")
        vds.exportVCF(f)
        hc.importVCF(f)
      }

      val actual = {
        val f = tmpDir.createTempFile(extension="vcf")
        truth.toVDS.exportVCF(f)
        hc.importVCF(f)
      }

      truth.same(actual)
    }.check()
  }

  @Test def notIdenticalHeaders() {
    val tmp1 = tmpDir.createTempFile("sample1", ".vcf")
    val tmp2 = tmpDir.createTempFile("sample2", ".vcf")
    val tmp3 = tmpDir.createTempFile("sample3", ".vcf")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val sampleIds = vds.sampleIds
    vds.filterSamplesList(Set(sampleIds(0))).exportVCF(tmp1)
    vds.filterSamplesList(Set(sampleIds(1))).exportVCF(tmp2)
    assert(intercept[SparkException] (hc.importVCFs(Array(tmp1, tmp2))).getMessage.contains("invalid sample ids"))

    vds.filterSamplesList(Set(sampleIds(0),sampleIds(1))).exportVCF(tmp3)
    assert(intercept[SparkException] (hc.importVCFs(Array(tmp1, tmp3))).getMessage.contains("invalid sample ids"))
  }
}
