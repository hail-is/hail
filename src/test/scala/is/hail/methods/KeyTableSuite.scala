package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.testng.annotations.Test

class KeyTableSuite extends SparkSuite {
  def sampleKT1: KeyTable = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))
    val keyNames = Array("Sample")

    KeyTable(hc, rdd, signature, keyNames)
  }

  def sampleKT2: KeyTable = {
    val data = Array(Array("Sample1", IndexedSeq(9, 1), 5), Array("Sample2", IndexedSeq(3), 5),
      Array("Sample3", IndexedSeq(2, 3, 4), 5), Array("Sample4", IndexedSeq.empty[Int], 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TArray(TInt32)), ("field2", TInt32))
    val keyNames = Array("Sample")
    KeyTable(hc, rdd, signature, keyNames)
  }

  def sampleKT3: KeyTable = {
    val data = Array(Array("Sample1", IndexedSeq(IndexedSeq(9, 10), IndexedSeq(1)), IndexedSeq(5, 6)), Array("Sample2", IndexedSeq(IndexedSeq(3), IndexedSeq.empty[Int]), IndexedSeq(5, 3)),
      Array("Sample3", IndexedSeq(IndexedSeq(2, 3, 4), IndexedSeq(3), IndexedSeq(4, 10)), IndexedSeq.empty[Int]), Array("Sample4", IndexedSeq.empty[Int], IndexedSeq(5)))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TArray(TArray(TInt32))), ("field2", TArray(TInt32)))
    val keyNames = Array("Sample")
    KeyTable(hc, rdd, signature, keyNames)
  }

  @Test def testImportExport() = {
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

  @Test def testAnnotate() = {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val kt1 = hc.importTable(inputFile, impute = true).keyBy("Sample")
    val kt2 = kt1.annotate("""qPhen2 = pow(qPhen, 2), NotStatus = Status == "CASE", X = qPhen == 5""")
    val kt3 = kt2.annotate("")
    val kt4 = kt3.select(kt3.fieldNames, Array("qPhen", "NotStatus"))

    val kt1FieldNames = kt1.fieldNames.toSet
    val kt2FieldNames = kt2.fieldNames.toSet

    assert(kt1.nKeys == 1)
    assert(kt2.nKeys == 1)
    assert(kt1.nFields == 3 && kt2.nFields == 6)
    assert(kt1.keyFields.zip(kt2.keyFields).forall { case (fd1, fd2) => fd1.name == fd2.name && fd1.typ == fd2.typ })
    assert(kt1FieldNames ++ Set("qPhen2", "NotStatus", "X") == kt2FieldNames)
    assert(kt2 same kt3)

    def getDataAsMap(kt: KeyTable) = {
      val fieldNames = kt.fieldNames
      val nFields = kt.nFields
      kt.rdd.map { a => fieldNames.zip(a.asInstanceOf[Row].toSeq).toMap }.collect().toSet
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
    val signature = TStruct(("field1", TInt32), ("field2", TInt32), ("field3", TInt32))
    val keyNames = Array("field1")

    val kt1 = KeyTable(hc, rdd, signature, keyNames)
    val kt2 = kt1.filter("field1 < 3", keep = true)
    val kt3 = kt1.filter("field1 < 3 && field3 == 4", keep = true)
    val kt4 = kt1.filter("field1 == 5 && field2 == 9 && field3 == 0", keep = false)
    val kt5 = kt1.filter("field1 < -5 && field3 == 100", keep = true)

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

    val nExpectedFields = ktLeft.nFields + ktRight.nFields - ktRight.nKeys

    val i: IndexedSeq[Int] = Array(1, 2, 3)

    val (_, leftKeyQuerier) = ktLeft.queryRow("Sample")
    val (_, rightKeyQuerier) = ktRight.queryRow("Sample")
    val (_, leftJoinKeyQuerier) = ktLeftJoin.queryRow("Sample")
    val (_, rightJoinKeyQuerier) = ktRightJoin.queryRow("Sample")

    val leftKeys = ktLeft.rdd.map { a => Option(leftKeyQuerier(a)).map(_.asInstanceOf[String]) }.collect().toSet
    val rightKeys = ktRight.rdd.map { a => Option(rightKeyQuerier(a)).map(_.asInstanceOf[String]) }.collect().toSet

    val nIntersectRows = leftKeys.intersect(rightKeys).size
    val nUnionRows = rightKeys.union(leftKeys).size
    val nExpectedKeys = ktLeft.nKeys

    assert(ktLeftJoin.count == ktLeft.count &&
      ktLeftJoin.nKeys == nExpectedKeys &&
      ktLeftJoin.nFields == nExpectedFields &&
      ktLeftJoin.filter { a =>
        !rightKeys.contains(Option(leftJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }.forall("isMissing(qPhen2) && isMissing(qPhen3)")
    )

    assert(ktRightJoin.count == ktRight.count &&
      ktRightJoin.nKeys == nExpectedKeys &&
      ktRightJoin.nFields == nExpectedFields &&
      ktRightJoin.filter { a =>
        !leftKeys.contains(Option(rightJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }.forall("isMissing(Status) && isMissing(qPhen)"))

    assert(ktOuterJoin.count == nUnionRows &&
      ktOuterJoin.nKeys == ktLeft.nKeys &&
      ktOuterJoin.nFields == nExpectedFields)

    assert(ktInnerJoin.count == nIntersectRows &&
      ktInnerJoin.nKeys == nExpectedKeys &&
      ktInnerJoin.nFields == nExpectedFields)

    val outputFile = tmpDir.createTempFile("join", "tsv")
    ktLeftJoin.export(outputFile)

    val noNull = ktLeft.filter("isDefined(qPhen) && isDefined(Status)", keep = true).keyBy(List("Sample", "Status"))
    assert(noNull.join(
      noNull.rename(Map("qPhen" -> "qPhen_")), "outer"
    ).rdd.forall { r => !r.toSeq.exists(_ == null)})
  }

  @Test def testJoinDiffKeyNames() = {
    val inputFile1 = "src/test/resources/sampleAnnotations.tsv"
    val inputFile2 = "src/test/resources/sampleAnnotations2.tsv"

    val ktLeft = hc.importTable(inputFile1, impute = true).keyBy("Sample")
    val ktRight = hc.importTable(inputFile2, impute = true)
      .keyBy("Sample")
      .rename(Map("Sample" -> "sample"))
    val ktBad = ktRight.select(ktRight.fieldNames, Array("qPhen2"))

    intercept[HailException] {
      val ktJoinBad = ktLeft.join(ktBad, "left")
      assert(ktJoinBad.key sameElements Array("Sample"))
    }

    val ktJoin = ktLeft.join(ktRight, "left")
    assert(ktJoin.key sameElements Array("Sample"))
  }

  @Test def testAggregate() {
    val data = Array(Array("Case", 9, 0), Array("Case", 3, 4), Array("Control", 2, 3), Array("Control", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("field1", TString), ("field2", TInt32), ("field3", TInt32))
    val keyNames = Array("field1")

    val kt1 = KeyTable(hc, rdd, signature, keyNames)
    val kt2 = kt1.aggregate("Status = field1",
      "A = field2.sum(), " +
        "B = field2.map(f => field2).sum(), " +
        "C = field2.map(f => field2 + field3).sum(), " +
        "D = field2.count(), " +
        "E = field2.filter(f => field2 == 3).count()"
    )

    kt2.export("test.tsv")
    val result = Array(Array("Case", 12, 12, 16, 2L, 1L), Array("Control", 3, 3, 11, 2L, 0L))
    val resRDD = sc.parallelize(result.map(Row.fromSeq(_)))
    val resSignature = TStruct(("Status", TString), ("A", TInt32), ("B", TInt32), ("C", TInt32), ("D", TInt64), ("E", TInt64))
    val ktResult = KeyTable(hc, resRDD, resSignature, key = Array("Status"))

    assert(kt2 same ktResult)

    val outputFile = tmpDir.createTempFile("aggregate", "tsv")
    kt2.export(outputFile)
  }

  @Test def testForallExists() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)
    assert(kt.forall("field2 == 5 && field1 != 0"))
    assert(!kt.forall("field2 == 0 && field1 == 5"))
    assert(kt.exists("""Sample == "Sample1" && field1 == 9 && field2 == 5"""))
    assert(!kt.exists("""Sample == "Sample1" && field1 == 13 && field2 == 2"""))
  }

  @Test def testRename() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)

    val rename1 = kt.rename(Array("ID1", "ID2", "ID3"))
    assert(rename1.fieldNames sameElements Array("ID1", "ID2", "ID3"))

    val rename2 = kt.rename(Map("field1" -> "ID1"))
    assert(rename2.fieldNames sameElements Array("Sample", "ID1", "field2"))

    intercept[HailException](kt.rename(Array("ID1")))

    intercept[HailException](kt.rename(Map("field1" -> "field2")))

    intercept[HailException](kt.rename(Map("Sample" -> "field2", "field1" -> "field2")))

    val outputFile = tmpDir.createTempFile("rename", "tsv")
    rename2.export(outputFile)
  }

  @Test def testSelect() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)

    val select1 = kt.select(Array("field1"), Array("field1"))
    assert((select1.key sameElements Array("field1")) && (select1.fieldNames sameElements Array("field1")))

    val select2 = kt.select(Array("Sample", "field2", "field1"), Array("Sample"))
    assert((select2.key sameElements Array("Sample")) && (select2.fieldNames sameElements Array("Sample", "field2", "field1")))

    val select3 = kt.select(Array("field2", "field1", "Sample"), Array.empty[String])
    assert((select3.key sameElements Array.empty[String]) && (select3.fieldNames sameElements Array("field2", "field1", "Sample")))

    val select4 = kt.select(Array.empty[String], Array.empty[String])
    assert((select4.key sameElements Array.empty[String]) && (select4.fieldNames sameElements Array.empty[String]))

    intercept[HailException](kt.select(Array.empty[String], Array("Sample")))
    intercept[HailException](kt.select(Array("Sample", "field2", "field5"), Array("Sample")))

    for (drop <- Array(select1, select2, select3, select4)) {
      drop.export(tmpDir.createTempFile("select", "tsv"))
    }
  }

  @Test def testDrop() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))

    val kt = KeyTable(hc, rdd, signature, Array("Sample"))
    val drop0 = kt.drop(Array.empty[String])
    assert((drop0.key sameElements Array("Sample")) && (drop0.fieldNames sameElements Array("Sample", "field1", "field2")))
    val drop1 = kt.drop(Array("Sample"))
    assert((drop1.key sameElements Array.empty[String]) && (drop1.fieldNames sameElements Array("field1", "field2")))
    val drop2 = kt.drop(Array("field1", "field2"))
    assert((drop2.key sameElements Array("Sample")) && (drop2.fieldNames sameElements Array("Sample")))
    val drop3 = kt.drop(Array("Sample", "field1"))
    assert((drop3.key sameElements Array.empty[String]) && (drop3.fieldNames sameElements Array("field2")))
    val drop4 = kt.drop(Array("Sample", "field2"))
    assert((drop4.key sameElements Array.empty[String]) && (drop4.fieldNames sameElements Array("field1")))
    val drop5 = kt.drop(Array("Sample", "field1", "field2"))
    assert((drop5.key sameElements Array.empty[String]) && (drop5.fieldNames sameElements Array.empty[String]))

    val kt2 = KeyTable(hc, rdd, signature, Array("field1", "field2"))
    val drop6 = kt2.drop(Array("field1"))
    assert((drop6.key sameElements Array("field2")) && (drop6.fieldNames sameElements Array("Sample", "field2")))

    for (drop <- Array(drop0, drop1, drop2, drop3, drop4, drop5, drop6)) {
      drop.export(tmpDir.createTempFile("drop", "tsv"))
    }

    intercept[HailException](kt.drop(Array("notInKT1", "notInKT2")))
  }

  @Test def testExplode() {
    val kt1 = sampleKT1
    val kt2 = sampleKT2
    val kt3 = sampleKT3

    val result2 = Array(Array("Sample1", 9, 5), Array("Sample1", 1, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5),
      Array("Sample3", 3, 5), Array("Sample3", 4, 5))
    val resRDD2 = sc.parallelize(result2.map(Row.fromSeq(_)))
    val ktResult2 = KeyTable(hc, resRDD2, TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32)), key = Array("Sample"))

    val result3 = Array(Array("Sample1", 9, 5), Array("Sample1", 10, 5), Array("Sample1", 9, 6), Array("Sample1", 10, 6),
      Array("Sample1", 1, 5), Array("Sample1", 1, 6), Array("Sample2", 3, 5), Array("Sample2", 3, 3))
    val resRDD3 = sc.parallelize(result3.map(Row.fromSeq(_)))
    val ktResult3 = KeyTable(hc, resRDD3, TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32)), key = Array("Sample"))

    intercept[HailException](kt1.explode(Array("Sample")))
    assert(ktResult2.same(kt2.explode(Array("field1"))))
    assert(ktResult3.same(kt3.explode(Array("field1", "field2", "field1"))))

    val outputFile = tmpDir.createTempFile("explode", "tsv")
    kt2.explode(Array("field1")).export(outputFile)
  }

  @Test def testKeyTableToDF() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    val kt = vds
      .variantsKT()
      .expandTypes()
      .flatten()
      .select(Array("va.info.MQRankSum"), Array.empty[String])

    val df = kt.toDF(sqlContext)
    df.printSchema()
    df.show()
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

    val Array(ktMean, ktStDev) = kt.query(Array("qPhen.stats().mean", "qPhen.stats().stdev")).map(_._1)

    assert(D_==(ktMean.asInstanceOf[Double], statComb.mean))
    assert(D_==(ktStDev.asInstanceOf[Double], statComb.stdev))

    val counter = localData.map(_.status).groupBy(identity).mapValues(_.length)

    val ktCounter = kt.query("Status.counter()")._1.asInstanceOf[Map[String, Long]]

    assert(ktCounter == counter)
  }

  @Test def test1725() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")
    val kt = vds.annotateVariantsExpr("va.id = str(v)")
      .variantsKT()
      .flatten()
      .select(Array("v", "va.id"), Array("va.id"))

    val kt2 = KeyTable(hc, vds.variants.map(v => Row(v.toString, 5)),
      TStruct("v" -> TString, "value" -> TInt32), Array("v"))

    kt.join(kt2, "inner").toDF(sqlContext).count()
  }

  @Test def testKeyOrder() {
    val kt1 = KeyTable(hc,
      sc.parallelize(Array(Row("foo", "bar", 3, "baz"))),
      TStruct(
        "f1" -> TString,
        "f2" -> TString,
        "f3" -> TInt32,
        "f4" -> TString
      ),
      Array("f3", "f2", "f1"))

    val kt2 = KeyTable(hc,
      sc.parallelize(Array(Row(3, "foo", "bar", "qux"))),
      TStruct(
        "f3" -> TInt32,
        "f1" -> TString,
        "f2" -> TString,
        "f5" -> TString
      ),
      Array("f3", "f2", "f1"))

    assert(kt1.join(kt2, "inner").count() == 1L)
    kt1.join(kt2, "outer").typeCheck()
  }

  @Test def testSame() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true)
    assert(kt.same(kt))
  }

  @Test def testSelfJoin() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true).keyBy("Sample")
    assert(kt.join(kt, "inner").forall("(isMissing(Status) || Status == Status_1) && " +
      "(isMissing(qPhen) || qPhen == qPhen_1)"))
  }
}
