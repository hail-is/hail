package is.hail.methods

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations._
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.{MatrixTable, VSMSubgen}
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.scalatest.Matchers.assert
import org.testng.annotations.Test

import scala.collection.mutable

class TableSuite extends SparkSuite {
  def sampleKT1: Table = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT2: Table = {
    val data = Array(Array("Sample1", IndexedSeq(9, 1), 5), Array("Sample2", IndexedSeq(3), 5),
      Array("Sample3", IndexedSeq(2, 3, 4), 5), Array("Sample4", IndexedSeq.empty[Int], 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TArray(TInt32())), ("field2", TInt32()))
    val keyNames = Array("Sample")
    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT3: Table = {
    val data = Array(Array("Sample1", IndexedSeq(IndexedSeq(9, 10), IndexedSeq(1)), IndexedSeq(5, 6)), Array("Sample2", IndexedSeq(IndexedSeq(3), IndexedSeq.empty[Int]), IndexedSeq(5, 3)),
      Array("Sample3", IndexedSeq(IndexedSeq(2, 3, 4), IndexedSeq(3), IndexedSeq(4, 10)), IndexedSeq.empty[Int]), Array("Sample4", IndexedSeq.empty[Int], IndexedSeq(5)))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TArray(TArray(TInt32()))), ("field2", TArray(TInt32())))
    val keyNames = Array("Sample")
    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  @Test def testImportExport() {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val outputFile = tmpDir.createTempFile("ktImpExp", "tsv")
    val kt = hc.importTable(inputFile).keyBy("Sample", "Status")
    kt.export(outputFile)

    val importedData = sc.hadoopConfiguration.readLines(inputFile)(_.map(_.value).toIndexedSeq)
    val exportedData = sc.hadoopConfiguration.readLines(outputFile)(_.map(_.value).toIndexedSeq)

    intercept[HailException] {
      hc.importTable(inputFile).keyBy(List("Sample", "Status", "BadKeyName"))
    }

    assert(importedData == exportedData)
  }

  @Test def testToMatrixTable() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val gkt = vds.entriesTable()

    val reVDS = gkt.toMatrixTable(Array("locus", "alleles"),
      Array("s"),
      vds.rowType.fieldNames.filter(x => x != "locus" && x != "alleles"),
      vds.colType.fieldNames.filter(_ != "s"),
      Array("locus"))

    val sampleOrder = vds.colKeys.toArray

    assert(reVDS.rowsTable().same(vds.rowsTable()))
    assert(reVDS.colsTable().same(vds.colsTable()))
    assert(reVDS.reorderCols(sampleOrder).same(vds))
  }

  @Test def testAnnotate() {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val kt1 = hc.importTable(inputFile, impute = true).keyBy("Sample")
    val kt2 = kt1.annotate("""qPhen2 = pow(row.qPhen.toFloat64, 2d), NotStatus = row.Status == "CASE", X = row.qPhen == 5""")
    val kt3 = kt2.annotate("")
    val kt4 = kt3.select(kt3.fieldNames.map(n => s"row.$n")).keyBy("qPhen", "NotStatus")

    val kt1columns = kt1.fieldNames.toSet
    val kt2columns = kt2.fieldNames.toSet

    assert(kt1.nKeys == 1)
    assert(kt2.nKeys == 1)
    assert(kt1.nColumns == 3 && kt2.nColumns == 6)
    assert(kt1.keyFields.zip(kt2.keyFields).forall { case (fd1, fd2) => fd1.name == fd2.name && fd1.typ == fd2.typ })
    assert(kt1columns ++ Set("qPhen2", "NotStatus", "X") == kt2columns)
    assert(kt2 same kt3)

    def getDataAsMap(kt: Table) = {
      val columns = kt.fieldNames
      kt.rdd.map { a => columns.zip(a.toSeq).toMap }.collect().toSet
    }

    val kt3data = getDataAsMap(kt3)
    val kt4data = getDataAsMap(kt4)

    assert(kt4.key.toSet == Set("qPhen", "NotStatus") &&
      kt4.fieldNames.toSet -- kt4.key == Set("qPhen2", "X", "Sample", "Status") &&
      kt3data == kt4data
    )

    val outputFile = tmpDir.createTempFile("annotate", "tsv")
    kt2.export(outputFile)
  }

  @Test def testFilter() = {
    val data = Array(Array(5, 9, 0), Array(2, 3, 4), Array(1, 2, 3))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TInt32()), ("field2", TInt32()), ("field3", TInt32()))
    val keyNames = Array("field1")

    val kt1 = Table(hc, rdd, signature, keyNames)
    kt1.typeCheck()
    val kt2 = kt1.filter("row.field1 < 3", keep = true)
    val kt3 = kt1.filter("row.field1 < 3 && row.field3 == 4", keep = true)
    val kt4 = kt1.filter("row.field1 == 5 && row.field2 == 9 && row.field3 == 0", keep = false)
    val kt5 = kt1.filter("row.field1 < -5 && row.field3 == 100", keep = true)

    assert(kt1.count == 3 && kt2.count == 2 && kt3.count == 1 && kt4.count == 2 && kt5.count == 0)

    val outputFile = tmpDir.createTempFile("filter", "tsv")
    kt5.export(outputFile)
  }

  @Test def testJoin() = {
    val inputFile1 = "src/test/resources/sampleAnnotations.tsv"
    val inputFile2 = "src/test/resources/sampleAnnotations2.tsv"

    val ktLeft = hc.importTable(inputFile1, impute = true).keyBy("Sample")
    val ktRight = hc.importTable(inputFile2, impute = true).keyBy("Sample")

    val ktLeftJoin = ktLeft.join(ktRight, "left")
    val ktRightJoin = ktLeft.join(ktRight, "right")
    val ktInnerJoin = ktLeft.join(ktRight, "inner")
    val ktOuterJoin = ktLeft.join(ktRight, "outer")

    val nExpectedColumns = ktLeft.nColumns + ktRight.nColumns - ktRight.nKeys

    val i: IndexedSeq[Int] = Array(1, 2, 3)

    val leftKeyQuerier = ktLeft.signature.query("Sample")
    val rightKeyQuerier = ktRight.signature.query("Sample")
    val leftJoinKeyQuerier = ktLeftJoin.signature.query("Sample")
    val rightJoinKeyQuerier = ktRightJoin.signature.query("Sample")

    val leftKeys = ktLeft.rdd.map { a => Option(leftKeyQuerier(a)).map(_.asInstanceOf[String]) }.collect().toSet
    val rightKeys = ktRight.rdd.map { a => Option(rightKeyQuerier(a)).map(_.asInstanceOf[String]) }.collect().toSet

    val nIntersectRows = leftKeys.intersect(rightKeys).size
    val nUnionRows = rightKeys.union(leftKeys).size
    val nExpectedKeys = ktLeft.nKeys

    assert(ktLeftJoin.count == ktLeft.count &&
      ktLeftJoin.nKeys == nExpectedKeys &&
      ktLeftJoin.nColumns == nExpectedColumns &&
      ktLeftJoin.copy(rdd = ktLeftJoin.rdd.filter { a =>
        !rightKeys.contains(Option(leftJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }).forall("isMissing(row.qPhen2) && isMissing(row.qPhen3)")
    )

    assert(ktRightJoin.count == ktRight.count &&
      ktRightJoin.nKeys == nExpectedKeys &&
      ktRightJoin.nColumns == nExpectedColumns &&
      ktRightJoin.copy(rdd = ktRightJoin.rdd.filter { a =>
        !leftKeys.contains(Option(rightJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }).forall("isMissing(row.Status) && isMissing(row.qPhen)"))

    assert(ktOuterJoin.count == nUnionRows &&
      ktOuterJoin.nKeys == ktLeft.nKeys &&
      ktOuterJoin.nColumns == nExpectedColumns)

    assert(ktInnerJoin.count == nIntersectRows &&
      ktInnerJoin.nKeys == nExpectedKeys &&
      ktInnerJoin.nColumns == nExpectedColumns)

    val outputFile = tmpDir.createTempFile("join", "tsv")
    ktLeftJoin.export(outputFile)

    val noNull = ktLeft.filter("isDefined(row.qPhen) && isDefined(row.Status)", keep = true).keyBy(List("Sample", "Status"))
    assert(noNull.join(noNull.rename(Map("qPhen" -> "qPhen_"), Map()), "outer").rdd.forall { r => !r.toSeq.contains(null) })
  }

  @Test def testJoinDiffKeyNames() = {
    val inputFile1 = "src/test/resources/sampleAnnotations.tsv"
    val inputFile2 = "src/test/resources/sampleAnnotations2.tsv"

    val ktLeft = hc.importTable(inputFile1, impute = true).keyBy("Sample")
    val ktRight = hc.importTable(inputFile2, impute = true).keyBy("Sample").rename(Map("Sample" -> "sample"), Map())
    val ktBad = ktRight.keyBy("qPhen2")

    intercept[Exception] {
      val ktJoinBad = ktLeft.join(ktBad, "left")
      assert(ktJoinBad.key sameElements Array("Sample"))
    }

    val ktJoin = ktLeft.join(ktRight, "left")
    assert(ktJoin.key sameElements Array("Sample"))
  }

  @Test def testAggregate() {
    val data = Array(Array("Case", 9, 0), Array("Case", 3, 4), Array("Control", 2, 3), Array("Control", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString()), ("field2", TInt32()), ("field3", TInt32()))
    val keyNames = Array("field1")

    val kt1 = Table(hc, rdd, signature, keyNames)
    kt1.typeCheck()
    val kt2 = kt1.aggregate("Status = row.field1",
      "A = AGG.map(r => r.field2).sum(), " +
        "B = AGG.map(r => r.field2).sum(), " +
        "C = AGG.map(r => r.field2 + r.field3).sum(), " +
        "D = AGG.count(), " +
        "E = AGG.filter(r => r.field2 == 3).count()"
    )

    kt2.export("test.tsv")
    val result = Array(Array("Case", 12, 12, 16, 2L, 1L), Array("Control", 3, 3, 11, 2L, 0L))
    val resRDD = sc.parallelize(result.map(Row.fromSeq(_)))
    val resSignature = TStruct(("Status", TString()), ("A", TInt32()), ("B", TInt32()), ("C", TInt32()), ("D", TInt64()), ("E", TInt64()))
    val ktResult = Table(hc, resRDD, resSignature, key = Array("Status"))
    ktResult.typeCheck()

    assert(kt2 same ktResult)

    val outputFile = tmpDir.createTempFile("aggregate", "tsv")
    kt2.export(outputFile)
  }

  @Test def testForallExists() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    assert(kt.forall("row.field2 == 5 && row.field1 != 0"))
    assert(!kt.forall("row.field2 == 0 && row.field1 == 5"))
    assert(kt.exists("""row.Sample == "Sample1" && row.field1 == 9 && row.field2 == 5"""))
    assert(!kt.exists("""row.Sample == "Sample1" && row.field1 == 13 && row.field2 == 2"""))
  }

  @Test def testSelect() {
    val data = Array(Array("Sample1", 9, 5, Row(5, "bunny")), Array("Sample2", 3, 5, Row(6, "hello")), Array("Sample3", 2, 5, null), Array("Sample4", 1, 5, null))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()), ("field3", TStruct(("A", TInt32()), ("B", TString()))))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()

    val select1 = kt.select(Array("row.field1")).keyBy("field1")
    assert((select1.key sameElements Array("field1")) && (select1.fieldNames sameElements Array("field1")))

    val select2 = kt.select(Array("row.Sample", "row.field2", "row.field1")).keyBy("Sample")
    assert((select2.key sameElements Array("Sample")) && (select2.fieldNames sameElements Array("Sample", "field2", "field1")))

    val select3 = kt.select(Array("row.field2", "row.field1", "row.Sample")).keyBy()
    assert((select3.key sameElements Array.empty[String]) && (select3.fieldNames sameElements Array("field2", "field1", "Sample")))

    val select4 = kt.select(Array.empty[String])
    assert((select4.key sameElements Array.empty[String]) && (select4.fieldNames sameElements Array.empty[String]))

    for (select <- Array(select1, select2, select3, select4)) {
      select.export(tmpDir.createTempFile("select", "tsv"))
    }

    TestUtils.interceptFatal("Invalid key")(kt.select(Array.empty[String]).keyBy("Sample"))
  }

  @Test def testExplode() {
    val kt1 = sampleKT1
    val kt2 = sampleKT2
    val kt3 = sampleKT3

    val result2 = Array(Array("Sample1", 9, 5), Array("Sample1", 1, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5),
      Array("Sample3", 3, 5), Array("Sample3", 4, 5))
    val resRDD2 = sc.parallelize(result2.map(Row.fromSeq(_)))
    val ktResult2 = Table(hc, resRDD2, TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32())), key = Array("Sample"))
    ktResult2.typeCheck()

    val result3 = Array(Array("Sample1", 9, 5), Array("Sample1", 10, 5), Array("Sample1", 9, 6), Array("Sample1", 10, 6),
      Array("Sample1", 1, 5), Array("Sample1", 1, 6), Array("Sample2", 3, 5), Array("Sample2", 3, 3))
    val resRDD3 = sc.parallelize(result3.map(Row.fromSeq(_)))
    val ktResult3 = Table(hc, resRDD3, TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32())), key = Array("Sample"))
    ktResult3.typeCheck()

    intercept[HailException](kt1.explode(Array("Sample")))
    assert(ktResult2.same(kt2.explode(Array("field1"))))
    assert(ktResult3.same(kt3.explode(Array("field1", "field2", "field1"))))

    val outputFile = tmpDir.createTempFile("explode", "tsv")
    kt2.explode(Array("field1")).export(outputFile)
  }

  @Test def testKeyTableToDF() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val kt = vds
      .rowsTable()
      .expandTypes()
      .flatten()
      .select(Array("row.`info.MQRankSum`"))
      .copy2(globalSignature = TStruct.empty())

    val df = kt.toDF(sqlContext)
    assert(Table.fromDF(hc, df).same(kt))
  }

  @Test def testKeyTableToDF2() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val kt = vds
      .rowsTable()
      .annotate("locus = str(row.locus), alleles = str(row.alleles), filters = row.filters.toArray()")
      .flatten()
      .copy2(globalSignature = TStruct.empty())

    val df = kt.toDF(sqlContext)
    val kt2 = Table.fromDF(hc, df, key = Array("locus", "alleles"))
    assert(kt2.same(kt))
  }

  @Test def testQuery() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true)

    case class LineData(sample: String, status: String, qPhen: Option[Int])

    val localData = hadoopConf.readLines("src/test/resources/sampleAnnotations.tsv") { lines =>
      lines.drop(1).map { l =>
        val Array(sample, caseStatus, qPhen) = l.value.split("\t")
        LineData(sample, caseStatus, if (qPhen != "NA") Some(qPhen.toInt) else None)
      }.toArray
    }

    val statComb = localData.flatMap { ld => ld.qPhen }
      .aggregate(new StatCounter())({ case (sc, i) => sc.merge(i) }, { case (sc1, sc2) => sc1.merge(sc2) })

    val IndexedSeq(ktMean, ktStDev) = kt.query(
      "[AGG.map(r => r.qPhen.toFloat64).stats().mean , " +
      "AGG.map(r => r.qPhen.toFloat64).stats().stdev]").asInstanceOf[IndexedSeq[Double]]

    assert(D_==(ktMean.asInstanceOf[Double], statComb.mean))
    assert(D_==(ktStDev.asInstanceOf[Double], statComb.stdev))

    val counter = localData.map(_.status).groupBy(identity).mapValues(_.length)

    val ktCounter = kt.query("AGG.map(r => r.Status).counter()").asInstanceOf[Map[String, Long]]

    assert(ktCounter == counter)
  }

  @Test def testKeyOrder() {
    val kt1 = Table(hc,
      sc.parallelize(Array(Row("foo", "bar", 3, "baz"))),
      TStruct(
        "f1" -> TString(),
        "f2" -> TString(),
        "f3" -> TInt32(),
        "f4" -> TString()
      ),
      Array("f3", "f2", "f1"))
    kt1.typeCheck()

    val kt2 = Table(hc,
      sc.parallelize(Array(Row(3, "foo", "bar", "qux"))),
      TStruct(
        "f3" -> TInt32(),
        "f1" -> TString(),
        "f2" -> TString(),
        "f5" -> TString()
      ),
      Array("f3", "f2", "f1"))
    kt2.typeCheck()

    assert(kt1.join(kt2, "inner").count() == 1L)
    kt1.join(kt2, "outer").typeCheck()
  }

  @Test def testSame() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true)
    assert(kt.same(kt))
  }

  @Test def testSelfJoin() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true).keyBy("Sample")
    assert(kt.join(kt, "inner").forall("(isMissing(row.Status) || row.Status == row.Status_1) && " +
      "(isMissing(row.qPhen) || row.qPhen == row.qPhen_1)"))
  }

  @Test def issue2231() {
    assert(Table.range(hc, 100)
      .annotate("j = 1.0, i = 1")
      .keyBy("i").join(Table.range(hc, 100), "inner")
      .signature.fields.map(f => (f.name, f.typ)).toSet
      ===
      Set(("index", TInt32()), ("i", TInt32()), ("j", TFloat64())))
  }

  @Test def testGlobalAnnotations() {
    val kt = Table.range(hc, 10)
      .annotateGlobalExpr("foo = [1,2,3]")
      .annotateGlobal(Map(5 -> "bar"), TDict(TInt32Optional, TStringOptional), "dict")
      .annotateGlobalExpr("another = global.foo[1]")

    assert(kt.filter("global.dict.get(row.index) == \"bar\"", true).count() == 1)
    assert(kt.annotate("baz = global.foo").forall("row.baz == [1,2,3]"))
    assert(kt.forall("global.foo == [1,2,3]"))
    assert(kt.exists("global.dict.get(row.index) == \"bar\""))

    val gkt = kt.aggregate("index = row.index", "x = AGG.map(r => global.dict.get(r.index)).collect()[0]")
    assert(gkt.exists("row.x == \"bar\""))
    assert(kt.select(Array("baz = global.dict.get(row.index)")).exists("row.baz == \"bar\""))

    val tmpPath = tmpDir.createTempFile(extension = "kt")
    kt.write(tmpPath)
    assert(hc.readTable(tmpPath).same(kt))
  }

  @Test def testFlatten() {
    val table = Table.range(hc, 1).annotate("x = 5").keyBy("x")
    table.flatten()
  }

  @Test def testMaximalIndependentSet() {
    val edges = List((0, 4), (0, 1), (0, 2), (1, 5), (1, 3), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    val edgeTable = Table
      .parallelize(SparkSuite.hc, edges.map { case (i: Int, j: Int) => Row(i.toLong, j.toLong) }.toIndexedSeq,
        TStruct("i" -> TInt64(), "j" -> TInt64()), Array.empty[String], None)
    val misTable = edgeTable.maximalIndependentSet("row.i", "row.j", true, None)

    val mis = misTable.collect().map(row => row.getLong(0))

    val maximalIndependentSets = List(Set(0, 6, 5, 3), Set(1, 4, 7, 2))
    // imperfect greedy algorithm may also give non-maximal independent sets
    val nonMaximalIndependentSets = List(Set(0, 7), Set(6, 1))

    assert(nonMaximalIndependentSets.contains(mis.toSet) || maximalIndependentSets.contains(mis.toSet))
    assert(misTable.signature == TStruct("node" -> TInt64())
      && misTable.key == IndexedSeq("node")
      && misTable.globalSignature == TStruct())
  }

  @Test def testMaximalIndependentSetWithTieBreakerAndStrings() {
    val isCase = Map("A" -> true, "C" -> true, "E" -> true, "G" -> true, "H" -> true)
    val edges = List(("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"))
      .map { case (strA, strB) =>
        ((strA, isCase.getOrElse(strA, false)), (strB, isCase.getOrElse(strB, false)))
      }

    val iStruct = TStruct("id" -> TString(), "isCase" -> TBoolean())

    val edgeTable = Table
      .parallelize(SparkSuite.hc, edges.map { case (i, j) => Row(Row(i._1, i._2), Row(j._1, j._2)) }.toIndexedSeq,
        TStruct("i" -> iStruct, "j" -> iStruct), Array.empty[String], None)

    edgeTable.typeCheck()

    val misTable = edgeTable.maximalIndependentSet("row.i", "row.j", true,
      Some("(if ((l.isCase && (!(r.isCase)))) (-1) " +
        "else (if (((!(l.isCase)) && r.isCase)) (1) " +
        "else (0))).toInt64()"))

    val mis = misTable.collect().map(row => (row.getStruct(0).get(0), row.getStruct(0).get(1))).toSet
    val expectedSets = List(Set("A", "C", "E", "G"), Set("A", "C", "E", "H")).map(set => set.map(str => (str, true)))
    assert(expectedSets.contains(mis))
  }

  @Test def testSelectGlobals() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true)
    val kt2 = kt.selectGlobal("{x: 5}").selectGlobal("{y: global.x}")
    assert(kt2.globalSignature == TStruct("y" -> TInt32()) && kt2.globals.value.asInstanceOf[Row].get(0) == 5)
  }

  @Test def testQueryIR() {
    Table.range(hc, 100).query("AGG.map(x => row.index + 4).sum()")
  }
}
