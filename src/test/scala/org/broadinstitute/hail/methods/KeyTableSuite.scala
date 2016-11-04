package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.utils._
import org.testng.annotations.Test

class KeyTableSuite extends SparkSuite {

  @Test def testSingleToPairRDD() = {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    var s = State(sc, sqlContext)
    s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample, Status", inputFile))
    val kt = s.ktEnv("kt1")
    val kt2 = KeyTable(KeyTable.toSingleRDD(kt.rdd, kt.nKeys, kt.nValues), kt.signature, kt.keyNames.toArray)

    assert(kt.rdd.fullOuterJoin(kt2.rdd).forall { case (k, (v1, v2)) =>
      val res = v1 == v2
      if (!res)
        println(s"k=$k v1=$v1 v2=$v2 res=${ v1 == v2 }")
      res
    })
  }

  @Test def testImportExport() = {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val outputFile = tmpDir.createTempFile("ktImpExp", "tsv")
    var s = State(sc, sqlContext)
    s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample, Status", inputFile))
    s = ExportKeyTable.run(s, Array("-n", "kt1", "-o", outputFile))

    val importedData = sc.hadoopConfiguration.readLines(inputFile)(_.map(_.value).toIndexedSeq)
    val exportedData = sc.hadoopConfiguration.readLines(outputFile)(_.map(_.value).toIndexedSeq)

    intercept[FatalException] {
      s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample, Status, BadKeyName", inputFile))
    }

    assert(importedData == exportedData)
  }

  @Test def testAnnotate() = {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    var s = State(sc, sqlContext)

    s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample", "--impute", inputFile))
    s = AnnotateKeyTableExpr.run(s, Array("-n", "kt1", "-d", "kt2", "-c", """qPhen2 = pow(qPhen, 2), NotStatus = Status == "CASE", X = qPhen == 5"""))
    s = AnnotateKeyTableExpr.run(s, Array("-n", "kt2", "-d", "kt3"))
    s = AnnotateKeyTableExpr.run(s, Array("-n", "kt3", "-d", "kt4", "-k", "qPhen, NotStatus"))

    val kt1 = s.ktEnv("kt1")
    val kt2 = s.ktEnv("kt2")
    val kt3 = s.ktEnv("kt3")
    val kt4 = s.ktEnv("kt4")

    val kt1ValueNames = kt1.valueNames.toSet
    val kt2ValueNames = kt2.valueNames.toSet

    assert(kt1.nKeys == kt2.nKeys &&
      kt1.nValues == 2 && kt2.nValues == 5 &&
      kt1.keySignature == kt2.keySignature &&
      kt1ValueNames ++ Set("qPhen2", "NotStatus", "X") == kt2ValueNames
    )

    assert(kt2 same kt3)

    def getDataAsMap(kt: KeyTable) = {
      val fieldNames = kt.fieldNames
      val nFields = kt.nFields
      KeyTable.toSingleRDD(kt.rdd, kt.nKeys, kt.nValues)
        .map { a => fieldNames.zip(KeyTable.annotationToSeq(a, nFields)).toMap }.collect().toSet
    }

    val kt3data = getDataAsMap(kt3)
    val kt4data = getDataAsMap(kt4)

    assert(kt4.keyNames.toSet == Set("qPhen", "NotStatus") &&
      kt4.valueNames.toSet == Set("qPhen2", "X", "Sample", "Status") &&
      kt3data == kt4data
    )
  }

  @Test def testFilter() = {
    val data = Array(Array(5, 9, 0), Array(2, 3, 4), Array(1, 2, 3))
    val rdd = sc.parallelize(data.map(Annotation.fromSeq(_)))
    val signature = TStruct(("field1", TInt), ("field2", TInt), ("field3", TInt))
    val keyNames = Array("field1")
    val kt = KeyTable(rdd, signature, keyNames)

    var s = State(sc, sqlContext, ktEnv = Map("kt1" -> kt))

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < 3", "-d", "kt2", "--keep"))
    assert(s.ktEnv.contains("kt2") && s.ktEnv("kt2").nRows == 2)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < 3 && field3 == 4", "-d", "kt3", "--keep"))
    assert(s.ktEnv.contains("kt3") && s.ktEnv("kt3").nRows == 1)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 == 5 && field2 == 9 && field3 == 0", "-d", "kt3", "--remove"))
    assert(s.ktEnv.contains("kt3") && s.ktEnv("kt3").nRows == 2)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < -5 && field3 == 100", "--keep"))
    assert(s.ktEnv.contains("kt1") && s.ktEnv("kt1").nRows == 0)
  }

  @Test def testJoin() = {
    val inputFile1 = "src/test/resources/sampleAnnotations.tsv"
    val inputFile2 = "src/test/resources/sampleAnnotations2.tsv"

    var s = State(sc, sqlContext)
    s = ImportKeyTable.run(s, Array("-n", "ktLeft", "-k", "Sample", "--impute", inputFile1))
    s = ImportKeyTable.run(s, Array("-n", "ktRight", "-k", "Sample", "--impute", inputFile2))

    s = JoinKeyTable.run(s, Array("-l", "ktLeft", "-r", "ktRight", "-d", "ktLeftJoin", "-t", "left"))
    s = JoinKeyTable.run(s, Array("-l", "ktLeft", "-r", "ktRight", "-d", "ktRightJoin", "-t", "right"))
    s = JoinKeyTable.run(s, Array("-l", "ktLeft", "-r", "ktRight", "-d", "ktInnerJoin", "-t", "inner"))
    s = JoinKeyTable.run(s, Array("-l", "ktLeft", "-r", "ktRight", "-d", "ktOuterJoin", "-t", "outer"))

    val ktLeft = s.ktEnv("ktLeft")
    val ktRight = s.ktEnv("ktRight")

    val ktLeftJoin = s.ktEnv("ktLeftJoin")
    val ktRightJoin = s.ktEnv("ktRightJoin")
    val ktInnerJoin = s.ktEnv("ktInnerJoin")
    val ktOuterJoin = s.ktEnv("ktOuterJoin")

    val nExpectedValues = ktLeft.nValues + ktRight.nValues

    val (_, leftKeyQuery) = ktLeft.query("Sample")
    val (_, rightKeyQuery) = ktRight.query("Sample")

    val leftKeys = ktLeft.rdd.map { case (k, v) => leftKeyQuery(k, v).map(_.asInstanceOf[String]) }.collect().toSet
    val rightKeys = ktRight.rdd.map { case (k, v) => rightKeyQuery(k, v).map(_.asInstanceOf[String]) }.collect().toSet

    val nIntersectRows = leftKeys.intersect(rightKeys).size
    val nUnionRows = rightKeys.union(leftKeys).size
    val nExpectedKeys = ktLeft.nKeys

    assert(ktLeftJoin.nRows == ktLeft.nRows &&
      ktLeftJoin.nKeys == nExpectedKeys &&
      ktLeftJoin.nValues == nExpectedValues)

    assert(ktRightJoin.nRows == ktRight.nRows &&
      ktRightJoin.nKeys == nExpectedKeys &&
      ktRightJoin.nValues == nExpectedValues)

    assert(ktOuterJoin.nRows == nUnionRows &&
      ktOuterJoin.nKeys == ktLeft.nKeys &&
      ktOuterJoin.nValues == nExpectedValues)

    assert(ktInnerJoin.nRows == nIntersectRows &&
      ktInnerJoin.nKeys == nExpectedKeys &&
      ktInnerJoin.nValues == nExpectedValues)
  }
}
