package is.hail.io

import is.hail.SparkSuite
import is.hail.expr.types.TString
import is.hail.io.index.{IndexReader, IndexWriter, LeafChild}
import org.testng.annotations.{DataProvider, Test}
import is.hail.utils._

class IndexSuite extends SparkSuite {
  val strings = Array(
    "bear", "cat", "deer", "dog",
    "lion", "mouse", "parrot", "quail",
    "rabbit", "raccoon", "rat", "raven",
    "skunk", "snail", "squirrel", "vole",
    "weasel", "whale", "yak", "zebra")

  def writeIndex(file: String, data: Array[String], attributes: Map[String, Any] = Map.empty[String, Any]) {
    val iw = IndexWriter(hc.hadoopConf, file, TString(), branchingFactor = 2, attributes)
    data.zipWithIndex.foreach { case (s, offset) =>
      iw += (s, offset)
    }
    iw.close()
  }

  @DataProvider(name = "elements")
  def data(): Array[Array[Array[String]]] = {
    (1 to strings.length).map(i => Array(strings.take(i))).toArray
  }

  @Test(dataProvider = "elements")
  def writeReadGivesSameAsInput(data: Array[String]) {
    val file = tmpDir.createTempFile("test", "idx")
    val attributes = Map("foo" -> true, "bar" -> 5)

    writeIndex(file, data, attributes)
    assert(hc.hadoopConf.getFileSize(file) != 0)

    val ir = new IndexReader(hc.hadoopConf, file)
    assert(ir.attributes == attributes)
    data.zipWithIndex.foreach { case (s, i) =>
      assert(ir.queryByIndex(i).key == s)
      assert(ir.queryByKeyAllMatches(s).toFastIndexedSeq == IndexedSeq(LeafChild(s, i)))
      assert(ir.queryByKey(s, greater = true, closed = true).contains(LeafChild(s, i)))
      assert(ir.queryByKey(s, greater = false, closed = true).contains(LeafChild(s, i)))

      if (i != data.length - 1)
        assert(ir.queryByKey(s, greater = true, closed = false).map(_.key).contains(data(i + 1)))
      else
        assert(ir.queryByKey(s, greater = true, closed = false).isEmpty)

      if (i != 0)
        assert(ir.queryByKey(s, greater = false, closed = false).map(_.key).contains(data(i - 1)))
      else
        assert(ir.queryByKey(s, greater = false, closed = false).isEmpty)
    }
    ir.close()
  }

  @Test def testEmptyKeys() {
    val file = tmpDir.createTempFile("empty", "idx")
    writeIndex(file, Array.empty[String])
    val ir = new IndexReader(hc.hadoopConf, file)
    assert(ir.queryByKey("moo", true, true).isEmpty)
    assert(ir.queryByKey("moo", true, false).isEmpty)
    assert(ir.queryByKey("moo", false, true).isEmpty)
    assert(ir.queryByKey("moo", false, false).isEmpty)
    assert(ir.queryByKeyAllMatches("moo").isEmpty)
    intercept[IllegalArgumentException](ir.queryByIndex(0L))
    ir.close()
  }

  @Test def testDuplicateKeys1() {
    val strings = Array(
      "bear", "cat", "cat", "cat",
      "cat", "cat", "cat", "dog")

    val file = tmpDir.createTempFile("dups1", "idx")
    writeIndex(file, strings)
    val ir = new IndexReader(hc.hadoopConf, file)
    assert(ir.queryByKeyAllMatches("bear").toFastIndexedSeq == IndexedSeq(LeafChild("bear", 0L)))
    assert(ir.queryByKeyAllMatches("cat").toFastIndexedSeq == (1 until 7).map(i => LeafChild("cat", i.toLong)))
    assert(ir.queryByKeyAllMatches("dog").toFastIndexedSeq == IndexedSeq(LeafChild("dog", 7L)))
    assert(ir.queryByKeyAllMatches("foo").length == 0)
    assert(ir.queryByKeyAllMatches("aardvark").length == 0)

    assert(ir.queryByKey("aardvark", greater = true, closed = true).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("bear", greater = true, closed = true).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("boar", greater = true, closed = true).contains(LeafChild("cat", 1L)))
    assert(ir.queryByKey("cat", greater = true, closed = true).contains(LeafChild("cat", 6L)))
    assert(ir.queryByKey("cow", greater = true, closed = true).contains(LeafChild("dog", 7L)))
    assert(ir.queryByKey("dog", greater = true, closed = true).contains(LeafChild("dog", 7L)))
    assert(ir.queryByKey("elk", greater = true, closed = true).isEmpty)

    assert(ir.queryByKey("aardvark", greater = true, closed = false).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("bear", greater = true, closed = false).contains(LeafChild("cat", 1L)))
    assert(ir.queryByKey("boar", greater = true, closed = false).contains(LeafChild("cat", 1L)))
    assert(ir.queryByKey("cat", greater = true, closed = false).contains(LeafChild("dog", 7L)))
    assert(ir.queryByKey("cow", greater = true, closed = false).contains(LeafChild("dog", 7L)))
    assert(ir.queryByKey("dog", greater = true, closed = false).isEmpty)
    assert(ir.queryByKey("elk", greater = true, closed = false).isEmpty)

    assert(ir.queryByKey("aardvark", greater = false, closed = true).isEmpty)
    assert(ir.queryByKey("bear", greater = false, closed = true).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("boar", greater = false, closed = true).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("cat", greater = false, closed = true).contains(LeafChild("cat", 1L)))
    assert(ir.queryByKey("cow", greater = false, closed = true).contains(LeafChild("cat", 6L)))
    assert(ir.queryByKey("dog", greater = false, closed = true).contains(LeafChild("dog", 7L)))
    assert(ir.queryByKey("elk", greater = false, closed = true).contains(LeafChild("dog", 7L)))

    assert(ir.queryByKey("aardvark", greater = false, closed = false).isEmpty)
    assert(ir.queryByKey("bear", greater = false, closed = false).isEmpty)
    assert(ir.queryByKey("boar", greater = false, closed = false).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("cat", greater = false, closed = false).contains(LeafChild("bear", 0L)))
    assert(ir.queryByKey("cow", greater = false, closed = false).contains(LeafChild("cat", 6L)))
    assert(ir.queryByKey("dog", greater = false, closed = false).contains(LeafChild("cat", 6L)))
    assert(ir.queryByKey("elk", greater = false, closed = false).contains(LeafChild("dog", 7L)))

    assert(ir.queryByKeyAllMatchesOffsets(Array("cat", "dog", "bear")).toFastIndexedSeq == strings.indices)
  }

  @Test def testDuplicateKeys2() {
    val strings = Array(
      "bear", "bear", "cat", "cat",
      "cat", "cat", "cat", "cat",
      "cat", "dog", "mouse", "mouse",
      "skunk", "skunk", "skunk", "whale",
      "whale", "zebra", "zebra", "zebra")

    val file = tmpDir.createTempFile("dups2", "idx")
    writeIndex(file, strings)
    val ir = new IndexReader(hc.hadoopConf, file)

    val uniqueStrings = strings.distinct
    uniqueStrings.foreach { s =>
      assert(ir.queryByKey(s, greater = false, closed = true).contains(LeafChild(s, strings.indexOf(s))))
      assert(ir.queryByKey(s, greater = true, closed = true).contains(LeafChild(s, strings.lastIndexOf(s))))

      if (strings.head == s)
        assert(ir.queryByKey(s, greater = false, closed = false).isEmpty)
      else {
        val idx = strings.indexOf(s) - 1
        assert(ir.queryByKey(s, greater = false, closed = false).contains(LeafChild(strings(idx), idx)))
      }

      if (strings.last == s)
        assert(ir.queryByKey(s, greater = true, closed = false).isEmpty)
      else {
        val idx = strings.lastIndexOf(s) + 1
        assert(ir.queryByKey(s, greater = true, closed = false).contains(LeafChild(strings(idx), idx)))
      }
    }
  }

  @Test def testIndex() {
    hc.indexBgen(Array("src/test/resources/random.bgen"))
  }
}
