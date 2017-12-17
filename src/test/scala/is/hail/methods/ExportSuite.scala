package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.{Parser, TFloat64, TInt64}
import is.hail.utils._
import is.hail.testUtils._
import org.testng.annotations.Test

import scala.io.Source

class ExportSuite extends SparkSuite {

  @Test def test() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")
    vds = SplitMulti(vds)
    vds = SampleQC(vds)

    val out = tmpDir.createTempFile("out", ".tsv")
    vds.samplesKT().select("Sample = s", "sa.qc.*").export(out)

    val sb = new StringBuilder()
    sb.tsvAppend(Array(1, 2, 3, 4, 5))
    assert(sb.result() == "1,2,3,4,5")

    sb.clear()
    sb.tsvAppend(5.124)
    assert(sb.result() == "5.12400e+00")

    val readBackAnnotated = vds.annotateSamplesTable(hc.importTable(out, types = Map("callRate" -> TFloat64(),
      "nCalled" -> TInt64(),
      "nNotCalled" -> TInt64(),
      "nHomRef" -> TInt64(),
      "nHet" -> TInt64(),
      "nHomVar" -> TInt64(),
      "nSNP" -> TInt64(),
      "nInsertion" -> TInt64(),
      "nDeletion" -> TInt64(),
      "nSingleton" -> TInt64(),
      "nTransition" -> TInt64(),
      "nTransversion" -> TInt64(),
      "nStar" -> TInt64(),
      "dpMean" -> TFloat64(),
      "dpStDev" -> TFloat64(),
      "gqMean" -> TFloat64(),
      "gqStDev" -> TFloat64(),
      "nNonRef" -> TInt64(),
      "rTiTv" -> TFloat64(),
      "rHetHomVar" -> TFloat64(),
      "rInsertionDeletion" -> TFloat64())).keyBy("Sample"),
      root = "sa.readBackQC")

    val (t, qcQuerier) = readBackAnnotated.querySA("sa.qc")
    val (t2, rbQuerier) = readBackAnnotated.querySA("sa.readBackQC")
    assert(t == t2)
    readBackAnnotated.sampleAnnotations.foreach { annotation =>
      t.valuesSimilar(qcQuerier(annotation), rbQuerier(annotation))
    }
  }

  @Test def testExportSamples() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf")
      .filterSamplesExpr("""s == "C469::HG02026""""))
    assert(vds.nSamples == 1)

    // verify exports localSamples
    val f = tmpDir.createTempFile("samples", ".tsv")
    vds.samplesKT().select("s").export(f, header = false)
    assert(sc.textFile(f).count() == 1)
  }

  @Test def testAllowedNames() {
    val f = tmpDir.createTempFile("samples", ".tsv")
    val f2 = tmpDir.createTempFile("samples", ".tsv")
    val f3 = tmpDir.createTempFile("samples", ".tsv")

    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
    vds.samplesKT().select("`S.A.M.P.L.E.ID` = s").export(f)
    vds.samplesKT().select("`$$$I_HEARD_YOU_LIKE!_WEIRD~^_CHARS****` = s", "ANOTHERTHING = s").export(f2)
    vds.samplesKT().select("`I have some spaces and tabs\\there` = s", "`more weird stuff here` = s").export(f3)
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
    var vds = hc.importVCF("src/test/resources/sample.vcf")
    vds = SplitMulti(vds)
    vds = SampleQC(vds)
    vds
      .samplesKT()
      .select("computation = 5 * (if (sa.qc.callRate < .95) 0 else 1)")
      .export(f)
  }

  @Test def testTypes() {
    val vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))
    val out = tmpDir.createTempFile("export", ".out")

    vds.variantsKT().export(out, typesFile = out + ".types")

    val types = Parser.parseAnnotationTypes(hadoopConf.readFile(out + ".types")(Source.fromInputStream(_).mkString))
    val readBack = vds.annotateVariantsTable(hc.importTable(out, types = types).keyBy("v"),
      root = "va")
    assert(vds.same(readBack))
  }
}
