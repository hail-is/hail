package is.hail.methods

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.testng.annotations.Test

import scala.collection.mutable

class KeyTableSuite extends SparkSuite {
  def sampleKT1: KeyTable = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT2: KeyTable = {
    val data = Array(Array("Sample1", IndexedSeq(9, 1), 5), Array("Sample2", IndexedSeq(3), 5),
      Array("Sample3", IndexedSeq(2, 3, 4), 5), Array("Sample4", IndexedSeq.empty[Int], 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TArray(TInt32)), ("field2", TInt32))
    val keyNames = Array("Sample")
    val kt = KeyTable(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT3: KeyTable = {
    val data = Array(Array("Sample1", IndexedSeq(IndexedSeq(9, 10), IndexedSeq(1)), IndexedSeq(5, 6)), Array("Sample2", IndexedSeq(IndexedSeq(3), IndexedSeq.empty[Int]), IndexedSeq(5, 3)),
      Array("Sample3", IndexedSeq(IndexedSeq(2, 3, 4), IndexedSeq(3), IndexedSeq(4, 10)), IndexedSeq.empty[Int]), Array("Sample4", IndexedSeq.empty[Int], IndexedSeq(5)))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TArray(TArray(TInt32))), ("field2", TArray(TInt32)))
    val keyNames = Array("Sample")
    val kt = KeyTable(hc, rdd, signature, keyNames)
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

  @Test def testAnnotate() {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val kt1 = hc.importTable(inputFile, impute = true).keyBy("Sample")
    val kt2 = kt1.annotate("""qPhen2 = pow(qPhen, 2), NotStatus = Status == "CASE", X = qPhen == 5""")
    val kt3 = kt2.annotate("")
    val kt4 = kt3.select(kt3.columns).keyBy("qPhen", "NotStatus")

    val kt1columns = kt1.columns.toSet
    val kt2columns = kt2.columns.toSet

    assert(kt1.nKeys == 1)
    assert(kt2.nKeys == 1)
    assert(kt1.nColumns == 3 && kt2.nColumns == 6)
    assert(kt1.keyFields.zip(kt2.keyFields).forall { case (fd1, fd2) => fd1.name == fd2.name && fd1.typ == fd2.typ })
    assert(kt1columns ++ Set("qPhen2", "NotStatus", "X") == kt2columns)
    assert(kt2 same kt3)

    def getDataAsMap(kt: KeyTable) = {
      val columns = kt.columns
      kt.rdd.map { a => columns.zip(a.toSeq).toMap }.collect().toSet
    }

    val kt3data = getDataAsMap(kt3)
    val kt4data = getDataAsMap(kt4)

    assert(kt4.key.toSet == Set("qPhen", "NotStatus") &&
      kt4.columns.toSet -- kt4.key == Set("qPhen2", "X", "Sample", "Status") &&
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
    kt1.typeCheck()
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

    val nExpectedColumns = ktLeft.nColumns + ktRight.nColumns - ktRight.nKeys

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
      ktLeftJoin.nColumns == nExpectedColumns &&
      ktLeftJoin.filter { a =>
        !rightKeys.contains(Option(leftJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }.forall("isMissing(qPhen2) && isMissing(qPhen3)")
    )

    assert(ktRightJoin.count == ktRight.count &&
      ktRightJoin.nKeys == nExpectedKeys &&
      ktRightJoin.nColumns == nExpectedColumns &&
      ktRightJoin.filter { a =>
        !leftKeys.contains(Option(rightJoinKeyQuerier(a)).map(_.asInstanceOf[String]))
      }.forall("isMissing(Status) && isMissing(qPhen)"))

    assert(ktOuterJoin.count == nUnionRows &&
      ktOuterJoin.nKeys == ktLeft.nKeys &&
      ktOuterJoin.nColumns == nExpectedColumns)

    assert(ktInnerJoin.count == nIntersectRows &&
      ktInnerJoin.nKeys == nExpectedKeys &&
      ktInnerJoin.nColumns == nExpectedColumns)

    val outputFile = tmpDir.createTempFile("join", "tsv")
    ktLeftJoin.export(outputFile)

    val noNull = ktLeft.filter("isDefined(qPhen) && isDefined(Status)", keep = true).keyBy(List("Sample", "Status"))
    assert(noNull.join(
      noNull.rename(Map("qPhen" -> "qPhen_")), "outer"
    ).rdd.forall { r => !r.toSeq.exists(_ == null) })
  }

  @Test def testJoinDiffKeyNames() = {
    val inputFile1 = "src/test/resources/sampleAnnotations.tsv"
    val inputFile2 = "src/test/resources/sampleAnnotations2.tsv"

    val ktLeft = hc.importTable(inputFile1, impute = true).keyBy("Sample")
    val ktRight = hc.importTable(inputFile2, impute = true)
      .keyBy("Sample")
      .rename(Map("Sample" -> "sample"))
    val ktBad = ktRight.select(ktRight.columns).keyBy("qPhen2")

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
    kt1.typeCheck()
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
    ktResult.typeCheck()

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
    kt.typeCheck()
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
    kt.typeCheck()

    val rename1 = kt.rename(Array("ID1", "ID2", "ID3"))
    assert(rename1.columns sameElements Array("ID1", "ID2", "ID3"))

    val rename2 = kt.rename(Map("field1" -> "ID1"))
    assert(rename2.columns sameElements Array("Sample", "ID1", "field2"))

    intercept[HailException](kt.rename(Array("ID1")))

    intercept[HailException](kt.rename(Map("field1" -> "field2")))

    intercept[HailException](kt.rename(Map("Sample" -> "field2", "field1" -> "field2")))

    val outputFile = tmpDir.createTempFile("rename", "tsv")
    rename2.export(outputFile)
  }

  @Test def testSelect() {
    val data = Array(Array("Sample1", 9, 5, Row(5, "bunny")), Array("Sample2", 3, 5, Row(6, "hello")), Array("Sample3", 2, 5, null), Array("Sample4", 1, 5, null))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32), ("field3", TStruct(("A", TInt32), ("B", TString))))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)
    kt.typeCheck()

    val select1 = kt.select("field1").keyBy("field1")
    assert((select1.key sameElements Array("field1")) && (select1.columns sameElements Array("field1")))

    val select2 = kt.select("Sample", "field2", "field1").keyBy("Sample")
    assert((select2.key sameElements Array("Sample")) && (select2.columns sameElements Array("Sample", "field2", "field1")))

    val select3 = kt.select("field2", "field1", "Sample").keyBy()
    assert((select3.key sameElements Array.empty[String]) && (select3.columns sameElements Array("field2", "field1", "Sample")))

    val select4 = kt.select()
    assert((select4.key sameElements Array.empty[String]) && (select4.columns sameElements Array.empty[String]))

    val select5 = kt.select(Array("field3.*", "A = field1 + field2"), qualifiedName = true)
    assert(select5.signature == TStruct(("field3.A", TInt32), ("field3.B", TString), ("A", TInt32)))

    val select6 = kt.select("field2", "X = field1 + field2", "Z = 5", "field1", "field3.*", "Q.R = true")
    assert(select6.signature == TStruct(("field2", TInt32), ("X", TInt32), ("Z", TInt32),
      ("field1", TInt32), ("A", TInt32), ("B", TString), ("Q.R", TBoolean)))

    for (select <- Array(select1, select2, select3, select4, select5, select6)) {
      select.export(tmpDir.createTempFile("select", "tsv"))
    }

    TestUtils.interceptFatal("Invalid key")(kt.select().keyBy("Sample"))
    TestUtils.interceptFatal("symbol `field5' not found")(kt.select("Sample", "field2", "field5").keyBy("Sample"))
    TestUtils.interceptFatal("Either rename manually or use the 'mangle' option to handle duplicates")(kt.select("field3.*", "A = field1 + field2"))
    TestUtils.interceptFatal("left-hand side required in annotation expression")(kt.select("field1 + field2"))
    TestUtils.interceptFatal("cannot splat non-struct type:")(kt.select("field2.*"))
  }

  @Test def testDrop() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32))

    val kt = KeyTable(hc, rdd, signature, Array("Sample"))
    kt.typeCheck()
    val drop0 = kt.drop(Array.empty[String])
    assert((drop0.key sameElements Array("Sample")) && (drop0.columns sameElements Array("Sample", "field1", "field2")))
    val drop1 = kt.drop(Array("Sample"))
    assert((drop1.key sameElements Array.empty[String]) && (drop1.columns sameElements Array("field1", "field2")))
    val drop2 = kt.drop(Array("field1", "field2"))
    assert((drop2.key sameElements Array("Sample")) && (drop2.columns sameElements Array("Sample")))
    val drop3 = kt.drop(Array("Sample", "field1"))
    assert((drop3.key sameElements Array.empty[String]) && (drop3.columns sameElements Array("field2")))
    val drop4 = kt.drop(Array("Sample", "field2"))
    assert((drop4.key sameElements Array.empty[String]) && (drop4.columns sameElements Array("field1")))
    val drop5 = kt.drop(Array("Sample", "field1", "field2"))
    assert((drop5.key sameElements Array.empty[String]) && (drop5.columns sameElements Array.empty[String]))

    val kt2 = KeyTable(hc, rdd, signature, Array("field1", "field2"))
    val drop6 = kt2.drop(Array("field1"))
    assert((drop6.key sameElements Array("field2")) && (drop6.columns sameElements Array("Sample", "field2")))

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
    ktResult2.typeCheck()

    val result3 = Array(Array("Sample1", 9, 5), Array("Sample1", 10, 5), Array("Sample1", 9, 6), Array("Sample1", 10, 6),
      Array("Sample1", 1, 5), Array("Sample1", 1, 6), Array("Sample2", 3, 5), Array("Sample2", 3, 3))
    val resRDD3 = sc.parallelize(result3.map(Row.fromSeq(_)))
    val ktResult3 = KeyTable(hc, resRDD3, TStruct(("Sample", TString), ("field1", TInt32), ("field2", TInt32)), key = Array("Sample"))
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
      .variantsKT()
      .expandTypes()
      .flatten()
      .select("`va.info.MQRankSum`")

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
      .select("v", "`va.id`")
      .keyBy("va.id")

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
    kt1.typeCheck()

    val kt2 = KeyTable(hc,
      sc.parallelize(Array(Row(3, "foo", "bar", "qux"))),
      TStruct(
        "f3" -> TInt32,
        "f1" -> TString,
        "f2" -> TString,
        "f5" -> TString
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
    assert(kt.join(kt, "inner").forall("(isMissing(Status) || Status == Status_1) && " +
      "(isMissing(qPhen) || qPhen == qPhen_1)"))
  }

  @Test def testUngroup() {
    // test KeyTable method
    val data1 = Array(
      Array("Sample1", 5, Row(23.0f, "rabbit"), Row(1, "foo")),
      Array("Sample1", 5, null, Row(1, "foo")))
    val sig1 = TStruct(
      ("field0", TString),
      ("field1", TInt32),
      ("field2", TStruct(("1", TFloat32), ("2", TString))),
      ("field3", TStruct(("1", TInt32), ("2", TString))))
    val kt1 = KeyTable(hc, sc.parallelize(data1.map(Row.fromSeq(_))), sig1)
    kt1.typeCheck()

    val ungroupedData1 = Array(
      Array("Sample1", 5, 1, "foo", 23.0f, "rabbit"),
      Array("Sample1", 5, 1, "foo", null, null))
    val ungroupedSig1 = TStruct(
      ("field0", TString),
      ("field1", TInt32),
      ("1", TInt32),
      ("2", TString),
      ("field2.1", TFloat32),
      ("field2.2", TString)
    )
    val ungroupedKt1 = KeyTable(hc, sc.parallelize(ungroupedData1.map(Row.fromSeq(_))), ungroupedSig1)
    ungroupedKt1.typeCheck()

    assert(kt1.ungroup("field3").ungroup("field2", mangle = true).same(ungroupedKt1))

    TestUtils.interceptFatal("Can only ungroup fields of type Struct, but found type"){ kt1.ungroup("field0") }
    TestUtils.interceptFatal("Struct does not have field with name"){ kt1.ungroup("nonExistentField") }
    TestUtils.interceptFatal("Either rename manually or use the 'mangle' option to handle duplicates"){ kt1.ungroup("field2").ungroup("field3") }

    // test ungroup/group gives same result
    val data2 = Array(Array(Row(23, 1)))
    val rdd2 = sc.parallelize(data2.map(Row.fromSeq(_)))
    val kt2 = KeyTable(hc, rdd2, TStruct(("A", TStruct(("c1", TInt32), ("c2", TInt32)))))
    kt2.typeCheck()
    assert(kt2.ungroup("A").group("A", Array("c1", "c2")).same(kt2))

    // test function registry method
    val data3 = Array(Array(Row(6, Row("hello"))))
    val sig3 = TStruct(("foo", TStruct(("a", TInt32), ("b", TStruct(("i", TString))))))
    val kt3 = KeyTable(hc, sc.parallelize(data3.map(Row.fromSeq(_))), sig3)
    kt3.typeCheck()

    val ungroupedData3 = Array(Array(Row(6, "hello")))
    val ungroupedSig3 = TStruct(("foo", TStruct(("a", TInt32), ("i", TString))))
    val ungroupedKt3 = KeyTable(hc, sc.parallelize(ungroupedData3.map(Row.fromSeq(_))), ungroupedSig3)
    ungroupedKt3.typeCheck()

    assert(kt3.annotate("foo = ungroup(foo, b, false)").same(ungroupedKt3))
    assert(!kt3.annotate("foo = ungroup(foo, b, false)").same(kt3.annotate("foo = ungroup(foo, b, true)")))

    TestUtils.interceptFatal("invalid arguments for method"){ kt3.annotate("foo = ungroup(foo)") }
    TestUtils.interceptFatal("expects a Struct argument in the first position"){ kt3.annotate("foo = ungroup(false, b, true)") }
    TestUtils.interceptFatal("Expected boolean argument in the third position, but found a"){ kt3.annotate("foo = ungroup(foo, b, b)") }
    TestUtils.interceptFatal("Expected struct field identifier in the second position, but found a"){ kt3.annotate("foo = ungroup(foo, 2, true)") }
    TestUtils.interceptFatal("Struct does not have field with name"){ kt3.annotate("foo = ungroup(foo, notexist, true)") }
  }

  @Test def testGroup() {
    val data = Array(
      Array("Sample1", Row(23.0f, "rabbit"), 9, 5),
      null)

    val rdd = sc.parallelize(data.map(d => if (d == null) null else Row.fromSeq(d)))
    val kt = KeyTable(hc, rdd, TStruct(
      ("Sample", TString),
      ("field0", TStruct(("1", TFloat32), ("2", TString))),
      ("field1", TInt32),
      ("field2", TInt32)), key = Array("Sample"))
    kt.typeCheck()

    assert(kt.group("dest", Array("Sample", "field0")).signature == TStruct(("field1", TInt32), ("field2", TInt32),
      ("dest", TStruct(("Sample", TString), ("field0", TStruct(("1", TFloat32), ("2", TString)))))))

    assert(kt.group("Sample", Array("Sample", "field0")).signature == TStruct(("field1", TInt32), ("field2", TInt32),
      ("Sample", TStruct(("Sample", TString), ("field0", TStruct(("1", TFloat32), ("2", TString)))))))

    TestUtils.interceptFatal("Struct does not have field with name"){ kt.group("foo", Array("nonExistentField")) }

    val data2 = Array(Array(Row("Sample1", 5)))
    val rdd2 = sc.parallelize(data2.map(Row.fromSeq(_)))
    val kt2 = KeyTable(hc, rdd2, TStruct(("foo", TStruct(("a", TString), ("b", TInt32)))))
    kt2.typeCheck()

    val dataExpected = Array(Array(Row("Sample1", 5), Row(Row(5, "Sample1"))))
    val sigExpected = TStruct(("foo", TStruct(("a", TString), ("b", TInt32))), ("X", TStruct(("a", TStruct(("b", TInt32), ("a", TString))))))
    val kt2Expected = KeyTable(hc, sc.parallelize(dataExpected.map(Row.fromSeq(_))), sigExpected)
    kt2Expected.typeCheck()

    assert(kt2.annotate("X = group(foo, a, b, a)").same(kt2Expected))

    TestUtils.interceptFatal("Duplicate"){ kt2.annotate("X = group(foo, a, b, b)") }
    TestUtils.interceptFatal("Expected struct field identifiers after the first position, but found a"){ kt2.annotate("X = group(foo, x, true)") }
    TestUtils.interceptFatal("too few arguments for method"){ kt2.annotate("X = group(foo, a)") }
    TestUtils.interceptFatal("expects a Struct argument in the first position"){ kt2.annotate("X = group(2, x, true)") }
    TestUtils.interceptFatal("Struct does not have field with name"){ kt2.annotate("X = group(foo, x, y)") }
  }

  @Test def issue2231() {
    assert(KeyTable.range(hc, 100)
      .annotate("j = 1.0, i = 1")
      .keyBy("i").join(KeyTable.range(hc, 100), "inner")
      .signature.fields.map(f => (f.name, f.typ)).toSet
      ===
      Set(("index", TInt32), ("i", TInt32), ("j", TFloat64)))
  }

  @Test def mis() {
    // prefer to remove nodes with higher index
    assert(KeyTable.range(hc, 10)
      .annotate("i = index, j = index + 10")
      .maximalIndependentSet("i", "j", Some("l - r"))
      .toSet
      ===
      Set(0,1,2,3,4,5,6,7,8,9))
  }
}
