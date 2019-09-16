package is.hail.io

import is.hail.HailSuite
import is.hail.annotations.Annotation
import is.hail.expr.types.physical.{PInt32, PString, PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.io.index._
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class IndexSuite extends HailSuite {
  val strings = Array(
    "bear", "cat", "deer", "dog",
    "lion", "mouse", "parrot", "quail",
    "rabbit", "raccoon", "rat", "raven",
    "skunk", "snail", "squirrel", "vole",
    "weasel", "whale", "yak", "zebra")

  val stringsWithDups = Array(
    "bear", "bear", "cat", "cat",
    "cat", "cat", "cat", "cat",
    "cat", "dog", "mouse", "mouse",
    "skunk", "skunk", "skunk", "whale",
    "whale", "zebra", "zebra", "zebra")

  val leafsWithDups = stringsWithDups.zipWithIndex.map { case (s, i) => LeafChild(s, i, Row()) }

  @DataProvider(name = "elements")
  def data(): Array[Array[Array[String]]] = {
    (1 to strings.length).map(i => Array(strings.take(i))).toArray
  }

  def writeIndex(file: String,
    data: Array[Any],
    annotations: Array[Annotation],
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int,
    attributes: Map[String, Any]) {
    val codecSpec = CodecSpec.default

    val leafType = LeafNodeBuilder.typ(keyType, annotationType)
    val leafCodec = codecSpec.makeCodecSpec2(leafType)
    val leafEnc = leafCodec.buildEncoder(leafType)

    val intType = InternalNodeBuilder.typ(keyType, annotationType)
    val intCodec = codecSpec.makeCodecSpec2(intType)
    val intEnc = intCodec.buildEncoder(intType)

    val iw = new IndexWriter(hc.sFS, file, keyType, annotationType, leafEnc, intEnc, leafType, intType, branchingFactor, attributes)
    data.zip(annotations).zipWithIndex.foreach { case ((s, a), offset) =>
      iw += (s, offset, a)
    }
    iw.close()
  }

  def writeIndex(file: String,
    data: Array[String],
    annotations: Array[Annotation],
    annotationType: Type,
    branchingFactor: Int = 2,
    attributes: Map[String, Any] = Map.empty[String, Any]): Unit =
    writeIndex(file, data.map(_.asInstanceOf[Any]), annotations, PString(), PType.canonical(annotationType), branchingFactor, attributes)

  @Test(dataProvider = "elements")
  def writeReadGivesSameAsInput(data: Array[String]) {
    val file = tmpDir.createTempFile("test", "idx")
    val attributes = Map("foo" -> true, "bar" -> 5)

    val a: (Int) => Annotation = (i: Int) => Row(i % 2 == 0)

    for (branchingFactor <- 2 to 5) {
      writeIndex(file,
        data,
        data.indices.map(i => a(i)).toArray,
        TStruct("a" -> TBoolean()),
        branchingFactor,
        attributes)
      assert(hc.sFS.getFileSize(file) != 0)

      val index = IndexReader(hc.sFS, file)

      assert(index.attributes == attributes)

      data.zipWithIndex.foreach { case (s, i) =>
        assert({
          val result = index.queryByIndex(i)
          result.key == s && result.annotation == a(i)
        })
      }

      index.close()
    }
  }

  @Test def testEmptyKeys() {
    val file = tmpDir.createTempFile("empty", "idx")
    writeIndex(file, Array.empty[String], Array.empty[Annotation], TStruct("a" -> TBoolean()), 2)
    assert(hc.sFS.getFileSize(file) != 0)
    val index = IndexReader(hc.sFS, file)
    intercept[IllegalArgumentException](index.queryByIndex(0L))
    assert(index.queryByKey("moo").isEmpty)
    assert(index.queryByInterval("bear", "cat", includesStart = true, includesEnd = true).isEmpty)
    index.close()
  }

  @Test def testLowerBound() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("lowerBound", "idx")
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(i => Row()).toArray, TStruct(required = true), branchingFactor)
      val index = IndexReader(hc.sFS, file)

      val n = stringsWithDups.length
      val f = { i: Int => stringsWithDups(i) }

      val expectedResult = Array(
        "aardvark" -> 0, "bear" -> 0, "cat" -> 2,
        "dog" -> 9, "elk" -> 10, "mouse" -> 10,
        "opossum" -> 12, "skunk" -> 12, "snail" -> 15,
        "whale" -> 15, "zebra" -> 17, "zoo" -> stringsWithDups.length
      )

      expectedResult.foreach { case (s, expectedIdx) =>
        assert(index.lowerBound(s) == expectedIdx) // test full b-tree search works
      }
    }
  }

  @Test def testUpperBound() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("upperBound", "idx")
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(i => Row()).toArray, TStruct(required = true), branchingFactor = 2)
      val index = IndexReader(hc.sFS, file)

      val n = stringsWithDups.length
      val f = { i: Int => stringsWithDups(i) }

      val expectedResult = Array(
        "aardvark" -> 0, "bear" -> 2, "cat" -> 9,
        "dog" -> 10, "elk" -> 10, "mouse" -> 12,
        "opossum" -> 12, "skunk" -> 15, "snail" -> 15,
        "whale" -> 17, "zebra" -> stringsWithDups.length,
        "zoo" -> stringsWithDups.length
      )

      expectedResult.foreach { case (s, expectedIdx) =>
        assert(index.upperBound(s) == expectedIdx) // test full b-tree search works
      }
    }
  }

  @Test def testRangeIterator() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("range", "idx")
      val a = { (i: Int) => Row() }
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(a).toArray, TStruct(required = true), branchingFactor)
      val index = IndexReader(hc.sFS, file)

      val bounds = stringsWithDups.indices.toArray.combinations(2).toArray
      bounds.foreach(b => index.iterator(b(0), b(1)).toArray sameElements leafsWithDups.slice(b(0), b(1)))

      assert(index.iterator.toArray sameElements stringsWithDups.zipWithIndex.map { case (s, i) => LeafChild(s, i, a(i)) })
    }
  }

  @Test def testQueryByKey() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("key", "idx")
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(i => Row()).toArray, TStruct(required = true), branchingFactor)
      val index = IndexReader(hc.sFS, file)

      val stringsNotInList = Array("aardvark", "crow", "elk", "otter", "zoo")
      assert(stringsNotInList.forall(s => index.queryByKey(s).isEmpty))

      val stringsInList = stringsWithDups.distinct
      assert(stringsInList.forall(s => index.queryByKey(s) sameElements leafsWithDups.filter(_.key == s)))
    }
  }

  @Test def testIntervalIterator() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("interval", "idx")
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(i => Row()).toArray, TStruct(required = true), branchingFactor)
      val index = IndexReader(hc.sFS, file)

      // intervals with endpoint in list
      assert(index.queryByInterval("bear", "bear", includesStart = true, includesEnd = true).toFastIndexedSeq == index.iterator(0, 2).toFastIndexedSeq)

      assert(index.queryByInterval("bear", "cat", includesStart = true, includesEnd = false).toFastIndexedSeq == index.iterator(0, 2).toFastIndexedSeq)
      assert(index.queryByInterval("bear", "cat", includesStart = true, includesEnd = true).toFastIndexedSeq == index.iterator(0, 9).toFastIndexedSeq)
      assert(index.queryByInterval("bear", "cat", includesStart = false, includesEnd = true).toFastIndexedSeq == index.iterator(2, 9).toFastIndexedSeq)
      assert(index.queryByInterval("bear", "cat", includesStart = false, includesEnd = false).toFastIndexedSeq == index.iterator(2, 2).toFastIndexedSeq)

      // intervals with endpoint(s) not in list
      assert(index.queryByInterval("cat", "snail", includesStart = true, includesEnd = false).toFastIndexedSeq == index.iterator(2, 15).toFastIndexedSeq)
      assert(index.queryByInterval("cat", "snail", includesStart = true, includesEnd = true).toFastIndexedSeq == index.iterator(2, 15).toFastIndexedSeq)
      assert(index.queryByInterval("aardvark", "cat", includesStart = true, includesEnd = true).toFastIndexedSeq == index.iterator(0, 9).toFastIndexedSeq)
      assert(index.queryByInterval("aardvark", "cat", includesStart = false, includesEnd = true).toFastIndexedSeq == index.iterator(0, 9).toFastIndexedSeq)

      // illegal interval queries
      intercept[IllegalArgumentException](index.queryByInterval("bear", "bear", includesStart = false, includesEnd = false).toFastIndexedSeq)
      intercept[IllegalArgumentException](index.queryByInterval("bear", "bear", includesStart = false, includesEnd = true).toFastIndexedSeq)
      intercept[IllegalArgumentException](index.queryByInterval("bear", "bear", includesStart = true, includesEnd = false).toFastIndexedSeq)
      intercept[IllegalArgumentException](index.queryByInterval("cat", "bear", includesStart = true, includesEnd = true).toFastIndexedSeq)

      val endPoints = (stringsWithDups.distinct ++ Array("aardvark", "boar", "elk", "oppossum", "snail", "zoo")).combinations(2)
      val ordering = TString().ordering

      for (bounds <- endPoints) {
        for (includesStart <- Array(true, false)) {
          for (includesEnd <- Array(true, false)) {
            if (Interval.isValid(ordering, bounds(0), bounds(1), includesStart, includesEnd)) {
              val lowerBoundIdx =
                if (includesStart) {
                  stringsWithDups.indexWhere(bounds(0) <= _) match {
                    case -1 => stringsWithDups.length
                    case x: Int => x
                  }
                }
                else
                  stringsWithDups.lastIndexWhere(bounds(0) >= _) + 1 // want last index at transition point where key is gteq and then want to exclude that point (add 1)

              val upperBoundIdx =
                if (includesEnd)
                  stringsWithDups.lastIndexWhere(bounds(1) >= _) + 1 // want last index at transition point and then want to include that value so add 1
                else
                  stringsWithDups.lastIndexWhere(bounds(1) > _) + 1 // want last index before transition point and then want to include that value so add 1

              assert(index.queryByInterval(bounds(0), bounds(1), includesStart, includesEnd).toFastIndexedSeq ==
                index.iterator(lowerBoundIdx, upperBoundIdx).toFastIndexedSeq)

              if (includesStart)
                assert(index.iterateFrom(bounds(0)).toFastIndexedSeq ==
                  leafsWithDups.slice(lowerBoundIdx, stringsWithDups.length).toFastIndexedSeq)
              if (!includesEnd)
                assert(index.iterateUntil(bounds(1)).toFastIndexedSeq ==
                  leafsWithDups.slice(0, upperBoundIdx).toFastIndexedSeq)
            } else
              intercept[IllegalArgumentException](index.queryByInterval(bounds(0), bounds(1), includesStart, includesEnd))
          }
        }
      }
    }
  }

  @Test def testIntervalIteratorWorksWithGeneralEndpoints() {
    for (branchingFactor <- 2 to 5) {
      val keyType = PStruct("a" -> PString(), "b" -> PInt32())
      val file = tmpDir.createTempFile("from", "idx")
      writeIndex(file,
        stringsWithDups.zipWithIndex.map { case (s, i) => Row(s, i) },
        stringsWithDups.indices.map(i => Row()).toArray,
        keyType,
        +PStruct(),
        branchingFactor,
        Map.empty)

      val leafChildren = stringsWithDups.zipWithIndex.map { case (s, i) => LeafChild(Row(s, i), i, Row()) }.toFastIndexedSeq

      val index = IndexReader(hc.sFS, file)
      assert(index.queryByInterval(Row("cat", 3), Row("cat", 5), includesStart = true, includesEnd = false).toFastIndexedSeq ==
        leafChildren.slice(3, 5))
      assert(index.queryByInterval(Row("cat"), Row("cat", 5), includesStart = true, includesEnd = false).toFastIndexedSeq ==
        leafChildren.slice(2, 5))
      assert(index.queryByInterval(Row(), Row(), includesStart = true, includesEnd = true).toFastIndexedSeq ==
        leafChildren)
      assert(index.queryByInterval(Row(), Row("cat"), includesStart = true, includesEnd = false).toFastIndexedSeq ==
        leafChildren.take(2))
      assert(index.queryByInterval(Row("zebra"), Row(), includesStart = true, includesEnd = true).toFastIndexedSeq ==
        leafChildren.takeRight(3))
    }
  }

  @Test def testIterateFromUntil() {
    for (branchingFactor <- 2 to 5) {
      val file = tmpDir.createTempFile("from", "idx")
      writeIndex(file, stringsWithDups, stringsWithDups.indices.map(i => Row()).toArray, TStruct(required = true), branchingFactor)
      val index = IndexReader(hc.sFS, file)

      val uniqueStrings = stringsWithDups.distinct ++ Array("aardvark", "crow", "elk", "otter", "zoo")
      uniqueStrings.foreach { s =>
        var start = stringsWithDups.indexWhere(s <= _)
        if (start == -1)
          start = stringsWithDups.length
        assert(index.iterateFrom(s).toFastIndexedSeq == leafsWithDups.slice(start, stringsWithDups.length).toFastIndexedSeq)

        val end = stringsWithDups.lastIndexWhere(s > _) + 1
        assert(index.iterateUntil(s).toFastIndexedSeq == leafsWithDups.slice(0, end).toFastIndexedSeq)
      }
    }
  }
}
