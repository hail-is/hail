package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.{Annotation, Region, RegionValue, RegionValueBuilder}
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
import is.hail.rvd.OrderedRVD
import is.hail.table.Table
import is.hail.variant.{GenomeReference, MatrixTable, VSMSubgen}
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
    val newMatrixType = MatrixType.fromParts(vsm.matrixType.globalType,
      vsm.matrixType.colKey,
      vsm.matrixType.colType,
      Array("row_id"),
      Array("row_id"),
      TStruct("row_id" -> TInt64()) ++ vsm.matrixType.rowType,
      vsm.matrixType.entryType)
    val partStarts = vsm.partitionStarts()
    val newRowType = newMatrixType.rvRowType
    val oldRowType = vsm.matrixType.rvRowType

    val indexedRDD = vsm.rvd.rdd.mapPartitionsWithIndex { case (i, it) =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      val partStart = partStarts(i)

      it.zipWithIndex.map { case (rv, idx) =>
        region2.clear()
        rv2b.start(newRowType)
        rv2b.startStruct()
        rv2b.addLong(partStart + idx)
        var i = 0
        while (i < oldRowType.size) {
          if (oldRowType.isFieldDefined(rv, i))
            rv2b.addRegionValue(oldRowType.fieldType(i), rv.region, oldRowType.loadField(rv, i))
          else
            rv2b.setMissing()
          i += 1
        }
        rv2b.endStruct()
        rv2.setOffset(rv2b.end())
        rv2
      }
    }
    new MatrixTable(hc, newMatrixType, vsm.globals, vsm.colValues, OrderedRVD(newMatrixType.orvdType, indexedRDD, None, None))
  }

  def reKeyCols(vsm: MatrixTable): MatrixTable = {
    vsm.copy2(colValues = Array.tabulate[Annotation](vsm.colValues.length){ i => Row("col" + i.toString) })
  }

  def renameColKeyField(vsm: MatrixTable): MatrixTable = {
    vsm.annotateSamplesExpr("col_id = sa.s").keyColsBy("col_id").selectCols("sa.col_id")
  }

  def dropRowAnnotations(vsm: MatrixTable, dropFields: Option[Array[String]]=None): MatrixTable =
    dropFields match {
      case Some(fields) =>
        val keep = vsm.rowType.fieldNames.filterNot { fn => fields.contains(fn) }
        val keepString = keep.map { f => s"`$f`: va.`$f`" }.mkString(",")
        vsm.annotateVariantsExpr(s"va = {$keepString}")
      case None => vsm.annotateVariantsExpr("va = {}")
    }

  def getVAFieldsAndTypes(vsm: MatrixTable): (Array[String], Array[Type]) = {
    (vsm.rowType.fieldNames, vsm.rowType.fieldType)
  }

  def exportImportableVds(vsm: MatrixTable, header: Boolean=true): String = {
    val path = tmpDir.createTempFile(extension = "txt")
    val exportStrs = vsm.rowType.fieldNames.map(x => s"${ prettyIdentifier(x) } = va.${ prettyIdentifier(x) }").mkString(",")
    val kt = vsm.makeKT(exportStrs, "`` = g.x", Array("v"))
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
      val vsm = LoadMatrix(hc, files, Some(Array("v")), Array(TString()), Some(Array("v")))
    }
    assert(e.getMessage.contains("invalid sample ids"))
  }

  @Test def testMissingVals() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Some(Array("v")), Array(TString()), Some(Array("v")))
      vsm.rvd.count()
    }
    assert(e.getMessage.contains("Incorrect number"))
  }

  @Test def testWithHeaderAndKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), None, vaTypes, Some(Array("v")), cellType = vsm.entryType)
      }
      (vsm, actual)
    }
  }

  @Test def testNoHeaderWithKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Some(vaNames), vaTypes, Some(Array("v")), cellType = vsm.entryType, noHeader=true)
      }
      (reKeyCols(vsm), actual)
    }
  }

  @Test def testWithHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), None, vaTypes, None, cellType = vsm.entryType)
      }
      (reKeyRows(vsm), actual)
    }
  }

  @Test def testNoHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Some(vaNames), vaTypes, None, cellType = vsm.entryType, noHeader=true)
      }
      (reKeyCols(reKeyRows(vsm)), actual)
    }
  }
}
