package is.hail.io

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.types.{TBoolean, TString, TStruct, Type}
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

  def writeIndex(file: String,
    data: Array[Annotation],
    annotations: Array[Annotation],
    keyType: Type,
    annotationType: Type,
    branchingFactor: Int = 2,
    attributes: Map[String, Any] = Map.empty[String, Any]) {
    val iw = new IndexWriter(hc.hadoopConf, file, keyType, annotationType, branchingFactor, attributes)
    data.zip(annotations).zipWithIndex.foreach { case ((s, a), offset) =>
      iw += (s, offset, a)
    }
    iw.close()
  }

  @Test(dataProvider = "elements")
  def writeReadGivesSameAsInput(data: Array[String]) {
    val file = tmpDir.createTempFile("test", "idx")
    val attributes = Map("foo" -> true, "bar" -> 5)

    for (branchingFactor <- 2 to 5) {
      writeIndex(file,
        data.map(_.asInstanceOf[Annotation]),
        data.indices.map(i => Row(i % 2 == 0)).toArray,
        TString(),
        TStruct("a" -> TBoolean()),
        branchingFactor,
        attributes)

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
    writeIndex(file, Array.empty[Annotation], Array.empty[Annotation], TString(), TStruct("a" -> TBoolean()), 2)
    assert(hc.hadoopConf.getFileSize(file) != 0)

    val ir = new IndexReader(hc.hadoopConf, file)
    intercept[IllegalArgumentException](ir.queryByIndex(0L))
    ir.close()
  }

  @Test def testLastKeyCorrect() {
    val file = tmpDir.createTempFile("last-key", "idx")
    writeIndex(file,
      strings.map(_.asInstanceOf[Annotation]),
      strings.indices.map(i => Row()).toArray,
      TString(),
      TStruct())
    val ir = new IndexReader(hc.hadoopConf, file)
    val node = ir.readInternalNode(ir.metadata.rootOffset)
    assert(node.children.length == 2 &&
      node.children(0).firstKey == "bear" && node.children(0).lastKey == "vole" &&
      node.children(1).firstKey == "weasel" && node.children(1).lastKey == "zebra")
  }
}
