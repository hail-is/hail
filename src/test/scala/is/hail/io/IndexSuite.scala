package is.hail.io

import is.hail.SparkSuite
import is.hail.expr.types.{TBoolean, TString, TStruct}
import is.hail.io.index.{IndexReader, IndexWriter}
import org.testng.annotations.{DataProvider, Test}
import is.hail.utils._
import org.apache.spark.sql.Row

class IndexSuite extends SparkSuite {
  val strings = Array(
    "bear", "cat", "deer", "dog",
    "lion", "mouse", "parrot", "quail",
    "rabbit", "raccoon", "rat", "raven",
    "skunk", "snail", "squirrel", "vole",
    "weasel", "whale", "yak", "zebra")

  @DataProvider(name = "elements")
  def data(): Array[Array[Array[String]]] = {
    (1 to strings.length).map(i => Array(strings.take(i))).toArray
  }

  @Test(dataProvider = "elements")
  def writeReadGivesSameAsInput(data: Array[String]) {
    val file = tmpDir.createTempFile("test", "idx")
    val attributes = Map("foo" -> true, "bar" -> 5)

    for (branchingFactor <- 2 to 5) {
      val iw = new IndexWriter(hc.hadoopConf, file, TString(), TStruct("a" -> TBoolean()), branchingFactor, attributes)
      data.zipWithIndex.foreach { case (s, offset) =>
        iw += (s, offset, Row(offset % 2 == 0))
      }
      iw.close()

      assert(hc.hadoopConf.getFileSize(file) != 0)

      val ir = new IndexReader(hc.hadoopConf, file)
      assert(ir.attributes == attributes)
      data.zipWithIndex.foreach { case (s, i) =>
        val result = ir.queryByIndex(i)
        assert(result.key == s && result.annotation == Row(i % 2 == 0))
      }
      ir.close()
    }
  }

  @Test def testEmptyKeys() {
    val file = tmpDir.createTempFile("empty", "idx")
    val iw = new IndexWriter(hc.hadoopConf, file, TString(), TStruct("a" -> TBoolean()), 2)
    iw.close()

    val ir = new IndexReader(hc.hadoopConf, file)
    intercept[IllegalArgumentException](ir.queryByIndex(0L))
    ir.close()
  }
}
