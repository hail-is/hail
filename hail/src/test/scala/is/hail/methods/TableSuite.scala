package is.hail.methods

import is.hail.{HailSuite, TestUtils}
import is.hail.expr._
import is.hail.expr.types.{virtual, _}
import is.hail.expr.types.virtual._
import is.hail.rvd.RVD
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class TableSuite extends HailSuite {
  def sampleKT1: Table = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = FastIndexedSeq("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT2: Table = {
    val data = Array(Array("Sample1", FastIndexedSeq(9, 1), 5), Array("Sample2", FastIndexedSeq(3), 5),
      Array("Sample3", FastIndexedSeq(2, 3, 4), 5), Array("Sample4", FastIndexedSeq.empty[Int], 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TArray(TInt32())), ("field2", TInt32()))
    val keyNames = FastIndexedSeq("Sample")
    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  def sampleKT3: Table = {
    val data = Array(
      Array("Sample1", FastIndexedSeq(FastIndexedSeq(9, 10), FastIndexedSeq(1)), FastIndexedSeq(5, 6)),
      Array("Sample2", FastIndexedSeq(FastIndexedSeq(3), FastIndexedSeq.empty[Int]), FastIndexedSeq(5, 3)),
      Array("Sample3", FastIndexedSeq(FastIndexedSeq(2, 3, 4), FastIndexedSeq(3), FastIndexedSeq(4, 10)),
        FastIndexedSeq.empty[Int]), Array("Sample4", FastIndexedSeq.empty[Int], FastIndexedSeq(5)))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TArray(TArray(TInt32()))), ("field2", TArray(TInt32())))
    val keyNames = FastIndexedSeq("Sample")
    val kt = Table(hc, rdd, signature, keyNames)
    kt.typeCheck()
    kt
  }

  @Test def testImportExport() {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val outputFile = tmpDir.createTempFile("ktImpExp", "tsv")
    val kt = hc.importTable(inputFile).keyBy(Array("Sample", "Status"))
    kt.export(outputFile)

    val importedData = sFS.readLines(inputFile)(_.map(_.value).toFastIndexedSeq)
    val exportedData = sFS.readLines(outputFile)(_.map(_.value).toFastIndexedSeq)

    intercept[AssertionError] {
      hc.importTable(inputFile).keyBy(Array("Sample", "Status", "BadKeyName"))
    }

    assert(importedData == exportedData)
  }

  @Test def testWriteReadOrdered() {
    val outputFile = tmpDir.createTempFile("ktRdWrtOrd")
    sampleKT1.write(outputFile)
    val read = Table.read(hc, outputFile)

    assert(read.same(sampleKT1))
  }

  @Test def testWriteReadUnordered() {
    val outputFile = tmpDir.createTempFile("ktRdWrtUnord")
    sampleKT1.unkey().write(outputFile)
    val read = Table.read(hc, outputFile)

    assert(read.same(sampleKT1.unkey()))
  }

  @Test def testKeyBy() {
    val kt = sampleKT1
    val count = kt.count()
    val kt2 = kt.keyBy(Array("Sample", "field1"))
    assert(kt2.count() == count)
    assert(kt2.keyBy(Array("Sample")).count() == count)
    assert(kt.keyBy(Array("field1")).count() == count)
    assert(kt.unkey().keyBy(Array("Sample")).count() == count)
  }

  @Test def testExplode() {
    val kt1 = sampleKT1
    val kt2 = sampleKT2
    val kt3 = sampleKT3

    val result2 = Array(Array("Sample1", 9, 5), Array("Sample1", 1, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5),
      Array("Sample3", 3, 5), Array("Sample3", 4, 5))
    val resRDD2 = sc.parallelize(result2.map(Row.fromSeq(_)))
    val ktResult2 = Table(hc, resRDD2,
      TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32())),
      key = FastIndexedSeq("Sample"))
    ktResult2.typeCheck()

    val result3 = Array(Array("Sample1", 9, 5), Array("Sample1", 10, 5), Array("Sample1", 9, 6), Array("Sample1", 10, 6),
      Array("Sample1", 1, 5), Array("Sample1", 1, 6), Array("Sample2", 3, 5), Array("Sample2", 3, 3))
    val resRDD3 = sc.parallelize(result3.map(Row.fromSeq(_)))
    val ktResult3 = Table(hc, resRDD3,
      TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32())),
      key = FastIndexedSeq("Sample"))
    ktResult3.typeCheck()

    assert(ktResult2.same(kt2.explode(Array("field1"))))
    assert(ktResult3.same(kt3.explode(Array("field1", "field2", "field1"))))

    val outputFile = tmpDir.createTempFile("explode", "tsv")
    kt2.explode(Array("field1")).export(outputFile)
  }

  @Test def testSame() {
    val kt = hc.importTable("src/test/resources/sampleAnnotations.tsv", impute = true)
    assert(kt.same(kt))
  }
}
