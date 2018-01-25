package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.variant.{GenomeReference, MatrixTable, VSMSubgen}
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._

class ImportMatrixSuite extends SparkSuite {

  val genMatrix = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Gen.const(TStruct.empty()),
    vSigGen = Gen.const(TString()),
    vaSigGen = Gen.const(TStruct.empty()),
    globalSigGen = Gen.const(TStruct.empty()),
    tSigGen = Gen.zip(Gen.oneOf[Type](TInt32(), TInt64(), TFloat32(), TFloat64(), TString()), Gen.coin(0.2))
      .map { case (typ, req) => typ.setRequired(req) }.map { t => TStruct("x" -> t) },
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genNonmissingValue,
    vaGen = (t: Type) => t.genNonmissingValue,
    globalGen = (t: Type) => t.genNonmissingValue,
    vGen = (t: Type) => Gen.identifier,
    tGen = (t: Type, v: Annotation) => t.genValue)

  @Test def testHeadersNotIdentical() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Some(Array("v")), Seq(TString()), Some("v"))
    }
    assert(e.getMessage.contains("invalid sample ids"))
  }

  @Test def testMissingVals() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Some(Array("v")), Seq(TString()), Some("v"))
      vsm.rvd.count()
    }
    assert(e.getMessage.contains("number of elements"))
  }

  // FIXME this test is broken, @tpoterba has fixed in a separate branch
  @Test(enabled = false) def testTypes() {
    forAll(MatrixTable.gen(hc, genMatrix)
      .filter(vsm => !vsm.stringSampleIds.contains("v"))) { vsm =>
      val actual: MatrixTable = {
        val f = tmpDir.createTempFile(extension = "txt")
        vsm.makeKT("v = va.v", "`` = g.x", Array("v")).export(f)
        LoadMatrix(hc, Array(f), None, Seq(TString()), Some("v"), cellType = vsm.entryType)
      }
      assert(vsm.unfilterEntries().same(actual
        .annotateSamplesExpr("s = sa.col_id")
        .keyColsBy("s")
        .selectCols("sa.s")
        .annotateVariantsExpr("v = va.v")
        .keyRowsBy(Array("v"), Array("v"))
        .selectRows("va.v"),
        // table impex outputs doubles with %.4e
        tolerance = 1e-4))


      val tmp1 = tmpDir.createTempFile(extension = "vds")
      vsm.write(tmp1, true)

      val vsm2 = MatrixTable.read(hc, tmp1)
      assert(vsm.same(vsm2))
      true
    }.check()
  }

  @Test def testTypesNoHeader() {
    forAll(MatrixTable.gen(hc, genMatrix)
      .filter(vsm => !vsm.sampleIds.contains("v"))) { vsm =>

      val actual2: MatrixTable = {
        val f = tmpDir.createTempFile(extension = "txt")
        vsm.makeKT("v = v", "`` = g.x", Array("v")).export(f, header=false)
        println(f)
        LoadMatrix(hc, Array(f), Some(Array("v")), Seq(TString()), Some("v"), cellType = vsm.entryType, noHeader=true)
      }
      assert(vsm.copy2(colValues = actual2.colValues).same(actual2.annotateVariantsExpr("va = {}")))


      val tmp1 = tmpDir.createTempFile(extension = "vds")
      vsm.write(tmp1, overwrite = true)

      val vsm2 = MatrixTable.read(hc, tmp1)
      assert(vsm.same(vsm2), "not same")
      true
    }.check()
  }

  @Test def testTypesNoKey() {
    forAll(MatrixTable.gen(hc, genMatrix)
      .filter(vsm => !vsm.sampleIds.contains("v"))) { vsm =>

      def getValuesKT(vsm: MatrixTable): Table = {
        vsm.renameSamples(vsm.sampleIds.zipWithIndex.map { case (id, i) =>
          "col" + i.toString }
          .toArray[Annotation])
          .makeKT("v = v", "`` = g.x", Array())
          .drop(Array("v"))
      }


      val actual2: MatrixTable = {
        val f = tmpDir.createTempFile(extension = "txt")
        getValuesKT(vsm).export(f, header=false)
        println(f)
        LoadMatrix(hc, Array(f), None, Seq(), None, cellType = vsm.genotypeSignature, noHeader=true)
      }
      println("loaded matrix!")
      val compare = vsm.annotateVariantsTable(vsm.variantsKT().index("idx"), Seq("v"), root="va").groupVariantsBy("va.idx", "x = gs.collect()[0].x")
      assert(compare.copy2(sampleIds = actual2.sampleIds).same(actual2), "not same")


      val tmp1 = tmpDir.createTempFile(extension = "vds")
      vsm.write(tmp1, true)

      val vsm2 = MatrixTable.read(hc, tmp1)
      assert(vsm.same(vsm2), "not same")
      true
    }.check()
  }
}
