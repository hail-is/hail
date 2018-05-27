package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
import is.hail.rvd.OrderedRVD
import is.hail.table.Table
import is.hail.variant.{ReferenceGenome$, MatrixTable, VSMSubgen}
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row

class ImportMatrixSuite extends SparkSuite {

  val genValidImportType = Gen.oneOf[Type](TInt32(), TInt64(), TFloat32(), TFloat64(), TString())

  val genImportableMatrix = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Gen.const(TStruct.empty()),
    vSigGen = Gen.oneOf[Type](TInt32(), TInt64(), TString()),
    rowPartitionKeyGen = (t: Type) => Gen.const(Array("v")),
    vaSigGen = Type.preGenStruct(required=false, genValidImportType),
    globalSigGen = Gen.const(TStruct.empty()),
    tSigGen = Gen.zip(genValidImportType, Gen.coin(0.2))
      .map { case (typ, req) => typ.setRequired(req) }.map { t => TStruct("x" -> t) },
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genNonmissingValue,
    vaGen = (t: Type) => t.genNonmissingValue,
    globalGen = (t: Type) => t.genNonmissingValue,
    vGen = (t: Type) => t.genNonmissingValue,
    tGen = (t: Type, v: Annotation) => t.genNonmissingValue)

  def reKeyRows(vsm: MatrixTable): MatrixTable = {
    val newFieldNames = "row_id" +: vsm.rowType.fieldNames
    vsm.indexRows("row_id").keyRowsBy(Array("row_id"), Array("row_id")).selectRows(s"{${ newFieldNames.map(n => s"$n: va.$n").mkString(",") }}", None)
  }

  def reKeyCols(vsm: MatrixTable): MatrixTable = {
    val rowMap = new java.util.HashMap[String, String](vsm.rowType.size)
    vsm.rowType.fields.foreach { f => rowMap.put(f.name, s"f${ f.index }") }

    val renamed = vsm.renameFields(rowMap,
      new java.util.HashMap[String, String](),
      new java.util.HashMap[String, String](),
      new java.util.HashMap[String, String]())

    renamed.copy2(colType = TStruct("s" -> TInt32()),
      colValues = vsm.colValues.copy(Array.tabulate[Annotation](vsm.colValues.value.length){ i => Row(i) }, t = TArray(TStruct("s" -> TInt32()))))
  }

  def renameColKeyField(vsm: MatrixTable): MatrixTable = {
    val colMap = new java.util.HashMap[String, String](vsm.rowType.size)
    colMap.put("s", "col_id")

    vsm.renameFields(new java.util.HashMap[String, String](),
      colMap,
      new java.util.HashMap[String, String](),
      new java.util.HashMap[String, String]())
  }

  def getVAFieldsAndTypes(vsm: MatrixTable): (Array[String], Array[Type]) = {
    (vsm.rowType.fieldNames, vsm.rowType.types)
  }

  def exportImportableVds(vsm: MatrixTable, header: Boolean=true): String = {
    val path = tmpDir.createTempFile(extension = "txt")
    val kt = vsm
      .selectEntries("{``: g.x}")
      .makeTable()
    kt.export(path, header=header)
    path
  }

  def checkValidResult(f: MatrixTable => (MatrixTable, MatrixTable)): Unit = {
    forAll(MatrixTable.gen(hc, genImportableMatrix)
      .filter(vsm => vsm.stringSampleIds.intersect(vsm.rowType.fieldNames).isEmpty)) { vsm =>
      val (transformed, result) = f(vsm)
      val transformed2 = renameColKeyField(transformed)
      assert(transformed2.same(result, tolerance=0.001))
      val tmp1 = tmpDir.createTempFile(extension = "vds")
      result.write(tmp1, true)

      val vsm2 = MatrixTable.read(hc, tmp1)
      assert(result.same(vsm2))
      true
    }.check()
  }

  @Test def testHeadersNotIdentical() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0"))
    }
    assert(e.getMessage.contains("invalid header"))
  }

  @Test def testMissingVals() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0"))
      vsm.rvd.count()
    }
    assert(e.getMessage.contains("Incorrect number"))
  }

  @Test def testWithHeaderAndKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Map(vaNames.zip(vaTypes): _*), Array("v"), cellType = vsm.entryType)
      }
      (vsm, actual)
    }
  }

  @Test def testNoHeaderWithKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        val newRowHeaders = Array.tabulate[String](vaNames.length)(i => s"f$i")
        LoadMatrix(hc, Array(f), Map(newRowHeaders.zip(vaTypes): _*), Array(s"f${ vaNames.indexOf("v") }"), cellType = vsm.entryType, noHeader=true)
      }
      (reKeyCols(vsm), actual)
    }
  }

  @Test def testWithHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Map(vaNames.zip(vaTypes): _*), Array(), cellType = vsm.entryType)
      }
      (reKeyRows(vsm), actual)
    }
  }

  @Test def testNoHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        val newRowHeaders = Array.tabulate[String](vaNames.length)(i => s"f$i")
        LoadMatrix(hc, Array(f), Map(newRowHeaders.zip(vaTypes): _*), Array(), cellType = vsm.entryType, noHeader=true)
      }
      (reKeyRows(reKeyCols(vsm)), actual)
    }
  }
}
