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
    val newMatrixType = vsm.matrixType.copy(vType = TInt64())
    val partStarts = vsm.partitionStarts()
    val newRowType = newMatrixType.rvRowType
    val oldRowType = vsm.matrixType.rvRowType

    val indexedRDD = vsm.rdd2.rdd.mapPartitionsWithIndex { case (i, it) =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      var idx = partStarts(i)

      it.map { rv =>
        region2.clear()
        rv2b.start(newRowType)
        rv2b.startStruct()
        rv2b.addLong(idx)
        rv2b.addLong(idx)
        rv2b.addRegionValue(oldRowType.fieldType(2), rv.region, oldRowType.loadField(rv, 2))
        rv2b.addRegionValue(oldRowType.fieldType(3), rv.region, oldRowType.loadField(rv, 3))
        rv2b.endStruct()
        idx += 1
        rv2.setOffset(rv2b.end())
        rv2
      }
    }
    new MatrixTable(hc, newMatrixType, vsm.value.localValue, OrderedRVD(newMatrixType.orderedRVType, indexedRDD, None, None))
  }

  def reKeyCols(vsm: MatrixTable): MatrixTable = {
    vsm.renameSamples(Array.tabulate[Annotation](vsm.nSamples)("col" + _.toString))
  }

  def dropRowAnnotations(vsm: MatrixTable, dropFields: Option[Array[String]]=None): MatrixTable =
    dropFields match {
      case Some(fields) =>
        val keep = vsm.vaSignature.fieldNames.filterNot { fn => fields.contains(fn) }
        val sb = new StringBuilder()
        keep.foreach { f =>
          sb ++= s"`$f`: va.`$f`,"
        }
        vsm.annotateVariantsExpr(s"va = {${ sb.result().dropRight(1) }}")
      case None => vsm.annotateVariantsExpr("va = {}")
    }

  def getVAFieldsAndTypes(vsm: MatrixTable): (Array[String], Array[Type]) = {
    (vsm.vaSignature.fieldNames, vsm.vaSignature.fieldType)
  }

  def exportImportableVds(vsm: MatrixTable, header: Boolean=true, rowKeys: Boolean=true): String = {
    val path = tmpDir.createTempFile(extension = "txt")
    val kt = if (rowKeys)
      vsm.makeKT("v = v, va.*", "`` = g.x", Array("v"))
    else
      vsm.makeKT("va.*", "`` = g.x", Array())
    kt.export(path, header=header)
    path
  }

  def checkValidResult(f: MatrixTable => (MatrixTable, MatrixTable)): Unit = {
    forAll(MatrixTable.gen(hc, genImportableMatrix)
      .filter(vsm => vsm.sampleIds.intersect(vsm.vaSignature.fieldNames).isEmpty &&
        !vsm.vaSignature.fieldNames.contains("v") &&
        !vsm.sampleIds.contains("v"))) { vsm =>
      println(vsm.sampleIds.contains("v"))
      val (transformed, result) = f(vsm)
      assert(transformed.same(result, tolerance=0.001))
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

  @Test def testWithHeaderAndKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), None, vsm.rowKeyTypes ++ vaTypes, Some("v"), cellType = vsm.entryType)
      }
      (vsm, dropRowAnnotations(actual, Some(Array("v"))))
    }
  }

  @Test def testNoHeaderWithKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Some(Array("v") ++ vaNames), vsm.rowKeyTypes ++ vaTypes, Some("v"), cellType = vsm.entryType, noHeader=true)
      }
      (reKeyCols(vsm), dropRowAnnotations(actual, Some(Array("v"))))
    }
  }

  @Test def testWithHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, rowKeys=false)
        val (_, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), None, vaTypes, None, cellType = vsm.entryType)
      }
      (reKeyRows(vsm), actual)
    }
  }

  @Test def testNoHeaderNoKey() {
    checkValidResult { vsm =>
      val actual: MatrixTable = {
        val f = exportImportableVds(vsm, header=false, rowKeys=false)
        val (vaNames, vaTypes) = getVAFieldsAndTypes(vsm)
        LoadMatrix(hc, Array(f), Some(vaNames), vaTypes, None, cellType = vsm.genotypeSignature, noHeader=true)
      }
      (reKeyCols(reKeyRows(vsm)), actual)
    }
  }
}
