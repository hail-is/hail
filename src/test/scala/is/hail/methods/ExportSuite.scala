package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.{Parser, TFloat64, TInt64}
import is.hail.utils._
import org.testng.annotations.Test

import scala.io.Source

class ExportSuite extends SparkSuite {

  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
      .sampleQC()

    val out = tmpDir.createTempFile("out", ".tsv")
    vds.exportSamples(out, "Sample = s, sa.qc.*")

    val sb = new StringBuilder()
    sb.tsvAppend(Array(1, 2, 3, 4, 5))
    assert(sb.result() == "1,2,3,4,5")

    sb.clear()
    sb.tsvAppend(5.124)
    assert(sb.result() == "5.12400e+00")

    val readBackAnnotated = vds.annotateSamplesTable(hc.importTable(out, types = Map("callRate" -> TFloat64,
      "nCalled" -> TInt64,
      "nNotCalled" -> TInt64,
      "nHomRef" -> TInt64,
      "nHet" -> TInt64,
      "nHomVar" -> TInt64,
      "nSNP" -> TInt64,
      "nInsertion" -> TInt64,
      "nDeletion" -> TInt64,
      "nSingleton" -> TInt64,
      "nTransition" -> TInt64,
      "nTransversion" -> TInt64,
      "nStar" -> TInt64,
      "dpMean" -> TFloat64,
      "dpStDev" -> TFloat64,
      "gqMean" -> TFloat64,
      "gqStDev" -> TFloat64,
      "nNonRef" -> TInt64,
      "rTiTv" -> TFloat64,
      "rHetHomVar" -> TFloat64,
      "rInsertionDeletion" -> TFloat64)).keyBy("Sample"),
      root = "sa.readBackQC")

    val (t, qcQuerier) = readBackAnnotated.querySA("sa.qc")
    val (t2, rbQuerier) = readBackAnnotated.querySA("sa.readBackQC")
    assert(t == t2)
    readBackAnnotated.sampleAnnotations.foreach { annotation =>
      t.valuesSimilar(qcQuerier(annotation), rbQuerier(annotation))
    }
  }

  @Test def testExportSamples() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
      .filterSamplesExpr("""s == "C469::HG02026"""")
    assert(vds.nSamples == 1)

    // verify exports localSamples
    val f = tmpDir.createTempFile("samples", ".tsv")
    vds.exportSamples(f, "s")
    assert(sc.textFile(f).count() == 1)
  }

  @Test def testAllowedNames() {
    val f = tmpDir.createTempFile("samples", ".tsv")
    val f2 = tmpDir.createTempFile("samples", ".tsv")
    val f3 = tmpDir.createTempFile("samples", ".tsv")

    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
    vds.exportSamples(f, "S.A.M.P.L.E.ID = s")
    vds.exportSamples(f2, "$$$I_HEARD_YOU_LIKE!_WEIRD~^_CHARS**** = s, ANOTHERTHING=s")
    vds.exportSamples(f3, "`I have some spaces and tabs\\there` = s,`more weird stuff here`=s")
    hadoopConf.readFile(f) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "S.A.M.P.L.E.ID")
    }
    hadoopConf.readFile(f2) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "$$$I_HEARD_YOU_LIKE!_WEIRD~^_CHARS****\tANOTHERTHING")
    }
    hadoopConf.readFile(f3) { reader =>
      val lines = Source.fromInputStream(reader)
        .getLines()
      assert(lines.next == "I have some spaces and tabs\there\tmore weird stuff here")
    }
  }

  @Test def testIf() {

    // this should run without errors
    val f = tmpDir.createTempFile("samples", ".tsv")
    hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
      .sampleQC()
      .exportSamples(f, "computation = 5 * (if (sa.qc.callRate < .95) 0 else 1)")
  }

  @Test def testTypes() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()
    val out = tmpDir.createTempFile("export", ".out")

    vds.exportVariants(out, "v = v, va = va", typeFile = true)

    val types = Parser.parseAnnotationTypes(hadoopConf.readFile(out + ".types")(Source.fromInputStream(_).mkString))
    val readBack = vds.annotateVariantsTable(hc.importTable(out, types=types).keyBy("v"),
      root = "va")
    assert(vds.same(readBack))
  }
}
