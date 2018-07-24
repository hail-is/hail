package is.hail.io

import is.hail.SparkSuite
import is.hail.expr.types.TString
import is.hail.io.index.{IndexReader, IndexWriter}
import org.testng.annotations.{DataProvider, Test}
import is.hail.utils._

class IndexSuite extends SparkSuite {
  val strings = Array(
    "bear", "cat", "deer", "dog",
    "lion", "mouse", "parrot", "quail",
    "rabbit", "raccoon", "rat", "raven",
    "skunk", "snail", "squirrel", "vole",
    "weasel", "whale", "yak", "zebra")

  @DataProvider(name = "elements")
  def data(): Array[Array[Array[String]]] = {
    (1 until strings.length).map(i => Array(strings.take(i))).toArray // FIXME: empty array ???
  }

  @Test(dataProvider = "elements")
  def writeReadGivesSameAsInput(data: Array[String]) {
    val file = tmpDir.createTempFile("test", "idx")

    val iw = new IndexWriter(hc.hadoopConf, file, TString(), branchingFactor = 2)
    data.zipWithIndex.foreach { case (s, offset) =>
      iw += (s, offset)
    }
    iw.close()

    assert(hc.hadoopConf.getFileSize(file) != 0)

    val ir = new IndexReader(hc.hadoopConf, file)
    data.zipWithIndex.foreach { case (s, i) =>
      assert(ir.queryByIndex(i).key == s)
    }
    ir.close()
  }
}
