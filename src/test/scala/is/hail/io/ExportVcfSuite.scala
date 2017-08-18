package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.expr.{TInt64, TStruct}
import is.hail.io.vcf.ExportVCF
import is.hail.utils._
import is.hail.variant.{Genotype, VSMSubgen, Variant, VariantSampleMatrix}
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps

class ExportVCFSuite extends SparkSuite {

  @Test def testSameAsOrigBGzip() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", ".vcf")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    assert(vdsOrig.same(hc.importVCF(outFile, nPartitions = Some(10)),
      tolerance = 1e-3))
  }

  @Test def testSameAsOrigNoCompression() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("export", ".vcf")
    val outFile2 = tmpDir.createTempFile("export2", ".vcf")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    val vdsNew = hc.importVCF(outFile, nPartitions = Some(10))

    assert(vdsOrig.same(vdsNew))

    val infoType = vdsNew.vaSignature.getAsOption[TStruct]("info").get
    val infoSize = infoType.size
    val toAdd = Annotation.fromSeq(Array.fill[Any](infoSize)(null))
    val (newVASignature, inserter) = vdsNew.insertVA(infoType, "info")

    val vdsNewMissingInfo = vdsNew.mapAnnotations(newVASignature,
      (v, va, gs) => inserter(va, toAdd))

    vdsNewMissingInfo.exportVCF(outFile2)

    assert(hc.importVCF(outFile2).same(vdsNewMissingInfo, 1e-2))
  }

  @Test def testSorted() {
    val vcfFile = "src/test/resources/multipleChromosomes.vcf"
    val outFile = tmpDir.createTempFile("sort", ".vcf.bgz")

    val vdsOrig = hc.importVCF(vcfFile, nPartitions = Some(10))

    vdsOrig.exportVCF(outFile)

    val vdsNew = hc.importVCF(outFile, nPartitions = Some(10))

    assert(hadoopConf.readFile(outFile) { s =>
      Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty && line(0) != '#')
        .map(line => line.split("\t")).take(5).map(a => Variant(a(0), a(1).toInt, a(3), a(4))).toArray
    }.isSorted)
  }

  @Test def testReadWrite() {

    val out = tmpDir.createTempFile("foo", ".vcf.bgz")
    val out2 = tmpDir.createTempFile("foo2", ".vcf.bgz")
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random), Gen.choose(1, 10),
      Gen.choose(1, 10)) { case (vds, nPar1, nPar2) =>
      hadoopConf.delete(out, recursive = true)
      hadoopConf.delete(out2, recursive = true)
      vds.exportVCF(out)
      val vds2 = hc.importVCF(out, nPartitions = Some(nPar1))
      vds.exportVCF(out2)
      hc.importVCF(out2, nPartitions = Some(nPar2)).same(vds2)
    }

    p.check()
  }

  @Test def testEmptyReadWrite() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").dropVariants()
    val out = tmpDir.createTempFile("foo", "vcf")
    val out2 = tmpDir.createTempFile("foo", "vcf.bgz")

    vds.exportVCF(out)
    vds.exportVCF(out2)

    assert(hadoopConf.getFileSize(out) > 0)
    assert(hadoopConf.getFileSize(out2) > 0)
    assert(hc.importVCF(out).same(vds))
    assert(hc.importVCF(out2).same(vds))
  }

  @Test def testGeneratedInfo() {
    val out = tmpDir.createTempFile("export", ".vcf")
    hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsExpr("va.info.AC = va.info.AC, va.info.another = 5")
      .exportVCF(out)

    hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(_.startsWith("##INFO"))
        .foreach { line =>
          assert(line.contains("Description="))
        }
    }
  }

  @Test def testCastLongToInt() {
    val out = tmpDir.createTempFile("cast", ".vcf")
    hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsExpr("va.info.AC_pass = gs.filter(g => g.gq >= 20 && g.dp >= 10 && (!g.isHet() || ( (g.ad[1]/g.ad.sum()) >= 0.2 ) )).count()")
      .exportVCF(out)

    hadoopConf.readFile(out) { in =>
      Source.fromInputStream(in)
        .getLines()
        .filter(l => l.startsWith("##INFO") && l.matches("AC_pass"))
        .foreach { line =>
          assert(line.contains("Type=Integer"))
        }
    }

    val out2 = tmpDir.createTempFile("cast2", ".vcf")
    hc.importVCF("src/test/resources/sample2.vcf")
      .annotateVariantsExpr("va.info.AC_pass = let x = 5.0 in x.toFloat64()")
      .exportVCF(out2)

    intercept[HailException] {
      val sb = new StringBuilder()
      ExportVCF.strVCF(sb, TInt64, 3147483647L)
    }
  }

  @Test def testErrors() {
    val out = tmpDir.createLocalTempFile("foo", "vcf")
    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      hc.importVCF("src/test/resources/sample2.vcf")
        .annotateVariantsExpr("va.info.foo = [[1]]")
        .exportVCF(out)
    }

    TestUtils.interceptFatal("INFO field 'foo': VCF does not support type") {
      hc.importVCF("src/test/resources/sample2.vcf")
        .annotateVariantsExpr("va.info.foo = v")
        .exportVCF(out)
    }
  }

  @Test def testInfoFieldSemicolons() {
    val vds = hc.importVCF("src/test/resources/sample.vcf", dropSamples = true)
      .annotateVariantsExpr("va.info = {foo: 5, bar: NA: Int}")

    val out = tmpDir.createLocalTempFile("foo", "vcf")
    vds.exportVCF(out)
    hadoopConf.readLines(out) { lines =>
      lines.foreach { l =>
        if (!l.value.startsWith("#")) {
          assert(l.value.contains("foo=5"))
          assert(!l.value.contains("foo=5;"))
        }
      }
    }
  }
}
