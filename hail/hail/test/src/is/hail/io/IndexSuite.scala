package is.hail.io

import is.hail.HailSuite
import is.hail.annotations.Annotation
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterator
import is.hail.io.index._
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PInt32, PType}
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row

class IndexSuite extends HailSuite {
  val strings = ArraySeq(
    "bear", "cat", "deer", "dog",
    "lion", "mouse", "parrot", "quail",
    "rabbit", "raccoon", "rat", "raven",
    "skunk", "snail", "squirrel", "vole",
    "weasel", "whale", "yak", "zebra")

  val stringsWithDups = ArraySeq(
    "bear", "bear", "cat", "cat",
    "cat", "cat", "cat", "cat",
    "cat", "dog", "mouse", "mouse",
    "skunk", "skunk", "skunk", "whale",
    "whale", "zebra", "zebra", "zebra")

  val leafsWithDups = stringsWithDups.zipWithIndex.map { case (s, i) =>
    LeafChild(s, i.toLong, Row())
  }

  def writeIndex(
    file: String,
    data: IndexedSeq[Any],
    annotations: IndexedSeq[Annotation],
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int,
    attributes: Map[String, Any],
  ): Unit = {
    val iw = IndexWriter.builder(ctx, keyType, annotationType, branchingFactor, attributes)(
      file,
      theHailClassLoader,
      ctx.taskContext,
      pool,
    )
    data.zip(annotations).zipWithIndex.foreach { case ((s, a), offset) =>
      iw.appendRow(s, offset.toLong, a)
    }
    iw.close()
  }

  def indexReader(file: String, annotationType: Type, keyPType: PType = PCanonicalString())
    : IndexReader =
    indexReader(file, PType.canonical(annotationType), keyPType)

  def indexReader(file: String, annotationPType: PType, keyPType: PType): IndexReader = {
    val leafPType = LeafNodeBuilder.typ(keyPType, annotationPType)
    val intPType = InternalNodeBuilder.typ(keyPType, annotationPType)
    val leafSpec = TypedCodecSpec(ctx, leafPType, BufferSpec.default)
    val intSpec = TypedCodecSpec(ctx, intPType, BufferSpec.default)

    val (lrt, leafDec) = leafSpec.buildDecoder(ctx, leafPType.virtualType)
    assert(lrt == leafPType)
    val (irt, intDec) = intSpec.buildDecoder(ctx, intPType.virtualType)
    assert(irt == intPType)
    IndexReaderBuilder.withDecoders(
      ctx,
      leafDec,
      intDec,
      keyPType.virtualType,
      annotationPType.virtualType,
      leafPType,
      intPType,
    ).apply(theHailClassLoader, fs, file, 8, pool)
  }

  def writeIndex(
    file: String,
    data: IndexedSeq[String],
    annotations: IndexedSeq[Annotation],
    annotationType: Type,
    branchingFactor: Int = 2,
    attributes: Map[String, Any] = Map.empty[String, Any],
  ): Unit =
    writeIndex(
      file,
      data.map(_.asInstanceOf[Any]),
      annotations,
      PCanonicalString(),
      PType.canonical(annotationType),
      branchingFactor,
      attributes,
    )

  object checkWriteReadGivesSameAsInput extends TestCases {
    def apply(
      data: IndexedSeq[String]
    )(implicit loc: munit.Location
    ): Unit = test("writeReadGivesSameAsInput") {
      val file = ctx.createTmpPath("test", "idx")
      val attributes: Map[String, Any] = Map("foo" -> true, "bar" -> 5)

      val a: (Int) => Annotation = (i: Int) => Row(i % 2 == 0)

      for (branchingFactor <- 2 to 5) {
        writeIndex(
          file,
          data,
          data.indices.map(i => a(i)),
          TStruct("a" -> TBoolean),
          branchingFactor,
          attributes,
        )
        assert(fs.getFileSize(file + "/index") != 0)
        assert(fs.getFileSize(file + "/metadata.json.gz") != 0)

        val index = indexReader(file, TStruct("a" -> TBoolean))

        assertEquals(index.attributes, attributes)

        data.zipWithIndex.foreach { case (s, i) =>
          val result = index.queryByIndex(i.toLong)
          assertEquals(result.key, s)
          assertEquals(result.annotation, a(i))
        }

        index.close()
      }
    }
  }

  (1 to strings.length).foreach(i => checkWriteReadGivesSameAsInput(strings.take(i)))

  test("EmptyKeys") {
    val file = ctx.createTmpPath("empty", "idx")
    writeIndex(file, ArraySeq.empty, ArraySeq.empty, TStruct("a" -> TBoolean), 2)
    assert(fs.getFileSize(file + "/index") != 0)
    assert(fs.getFileSize(file + "/metadata.json.gz") != 0)
    val index = indexReader(file, TStruct("a" -> TBoolean))
    intercept[IllegalArgumentException](index.queryByIndex(0L)): Unit
    assert(index.queryByKey("moo").isEmpty)
    assert(index.queryByInterval("bear", "cat", includesStart = true, includesEnd = true).isEmpty)
    index.close()
  }

  test("LowerBound") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("lowerBound", "idx")
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(i => Row()),
        TStruct.empty,
        branchingFactor,
      )
      val index = indexReader(file, TStruct.empty)

      val expectedResult = Array(
        "aardvark" -> 0,
        "bear" -> 0,
        "cat" -> 2,
        "dog" -> 9,
        "elk" -> 10,
        "mouse" -> 10,
        "opossum" -> 12,
        "skunk" -> 12,
        "snail" -> 15,
        "whale" -> 15,
        "zebra" -> 17,
        "zoo" -> stringsWithDups.length,
      )

      expectedResult.foreach { case (s, expectedIdx) =>
        assertEquals(index.lowerBound(s), expectedIdx.toLong)
      }
    }
  }

  test("UpperBound") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("upperBound", "idx")
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(i => Row()),
        TStruct.empty,
        branchingFactor = 2,
      )
      val index = indexReader(file, TStruct.empty)

      val expectedResult = Array(
        "aardvark" -> 0,
        "bear" -> 2,
        "cat" -> 9,
        "dog" -> 10,
        "elk" -> 10,
        "mouse" -> 12,
        "opossum" -> 12,
        "skunk" -> 15,
        "snail" -> 15,
        "whale" -> 17,
        "zebra" -> stringsWithDups.length,
        "zoo" -> stringsWithDups.length,
      )

      expectedResult.foreach { case (s, expectedIdx) =>
        assertEquals(index.upperBound(s), expectedIdx.toLong)
      }
    }
  }

  test("RangeIterator") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("range", "idx")
      val a = { (i: Int) => Row() }
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(a),
        TStruct.empty,
        branchingFactor,
      )
      val index = indexReader(file, TStruct.empty)

      val bounds = stringsWithDups.indices.combinations(2)
      bounds.foreach(b =>
        index.iterator(b(0).toLong, b(1).toLong) ==
          leafsWithDups.slice(b(0), b(1))
      )

      assert(index.iterator sameElements stringsWithDups.iterator.zipWithIndex.map { case (s, i) =>
        LeafChild(s, i.toLong, a(i))
      })
    }
  }

  test("QueryByKey") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("key", "idx")
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(i => Row()),
        TStruct.empty,
        branchingFactor,
      )
      val index = indexReader(file, TStruct.empty)

      val stringsNotInList = Array("aardvark", "crow", "elk", "otter", "zoo")
      assert(stringsNotInList.forall(s => index.queryByKey(s).isEmpty))

      val stringsInList = stringsWithDups.distinct
      assert(stringsInList.forall(s =>
        index.queryByKey(s) == leafsWithDups.filter(_.key == s)
      ))
    }
  }

  test("IntervalIterator") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("interval", "idx")
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(i => Row()),
        TStruct.empty,
        branchingFactor,
      )
      val index = indexReader(file, TStruct.empty)
      // intervals with endpoint in list
      assertEquals(
        index.queryByInterval(
          "bear",
          "bear",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 2).toFastSeq,
      )

      assertEquals(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(0, 2).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 9).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = false,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(2, 9).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = false,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(2, 2).toFastSeq,
      )

      // intervals with endpoint(s) not in list
      assertEquals(
        index.queryByInterval(
          "cat",
          "snail",
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(2, 15).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "cat",
          "snail",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(2, 15).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "aardvark",
          "cat",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 9).toFastSeq,
      )
      assertEquals(
        index.queryByInterval(
          "aardvark",
          "cat",
          includesStart = false,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 9).toFastSeq,
      )

      // illegal interval queries
      intercept[IllegalArgumentException](index.queryByInterval(
        "bear",
        "bear",
        includesStart = false,
        includesEnd = false,
      ).toFastSeq): Unit
      intercept[IllegalArgumentException](index.queryByInterval(
        "bear",
        "bear",
        includesStart = false,
        includesEnd = true,
      ).toFastSeq): Unit
      intercept[IllegalArgumentException](index.queryByInterval(
        "bear",
        "bear",
        includesStart = true,
        includesEnd = false,
      ).toFastSeq): Unit
      intercept[IllegalArgumentException](index.queryByInterval(
        "cat",
        "bear",
        includesStart = true,
        includesEnd = true,
      ).toFastSeq): Unit

      val endPoints = (stringsWithDups.distinct ++ Array("aardvark", "boar", "elk", "oppossum",
        "snail", "zoo")).combinations(2)
      val ordering = TString.ordering(ctx.stateManager)

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
                } else
                  stringsWithDups.lastIndexWhere(bounds(0) >= _) + 1 // want last index at transition point where key is gteq and then want to exclude that point (add 1)

              val upperBoundIdx =
                if (includesEnd)
                  stringsWithDups.lastIndexWhere(bounds(1) >= _) + 1 // want last index at transition point and then want to include that value so add 1
                else
                  stringsWithDups.lastIndexWhere(bounds(1) > _) + 1 // want last index before transition point and then want to include that value so add 1

              assertEquals(
                index.queryByInterval(
                  bounds(0),
                  bounds(1),
                  includesStart,
                  includesEnd,
                ).toFastSeq,
                index.iterator(lowerBoundIdx.toLong, upperBoundIdx.toLong).toFastSeq,
              )

              if (includesStart)
                assertEquals(
                  index.iterateFrom(bounds(0)).toFastSeq,
                  leafsWithDups.slice(lowerBoundIdx, stringsWithDups.length),
                )
              if (!includesEnd)
                assertEquals(
                  index.iterateUntil(bounds(1)).toFastSeq,
                  leafsWithDups.slice(0, upperBoundIdx),
                )
            } else
              intercept[IllegalArgumentException](index.queryByInterval(
                bounds(0),
                bounds(1),
                includesStart,
                includesEnd,
              ))
          }
        }
      }
    }
  }

  test("IntervalIteratorWorksWithGeneralEndpoints") {
    for (branchingFactor <- 2 to 5) {
      val keyType = PCanonicalStruct("a" -> PCanonicalString(), "b" -> PInt32())
      val file = ctx.createTmpPath("from", "idx")
      writeIndex(
        file,
        stringsWithDups.zipWithIndex.map { case (s, i) => Row(s, i) },
        stringsWithDups.indices.map(i => Row()),
        keyType,
        +PCanonicalStruct(),
        branchingFactor,
        Map.empty,
      )

      val leafChildren = stringsWithDups.zipWithIndex.map { case (s, i) =>
        LeafChild(Row(s, i), i.toLong, Row())
      }

      val index = indexReader(
        file,
        +PCanonicalStruct(),
        keyPType = PCanonicalStruct("a" -> PCanonicalString(), "b" -> PInt32()),
      )
      assertEquals(
        index.queryByInterval(
          Row("cat", 3),
          Row("cat", 5),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.slice(3, 5),
      )
      assertEquals(
        index.queryByInterval(
          Row("cat"),
          Row("cat", 5),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.slice(2, 5),
      )
      assertEquals(
        index.queryByInterval(
          Row(),
          Row(),
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        leafChildren,
      )
      assertEquals(
        index.queryByInterval(
          Row(),
          Row("cat"),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.take(2),
      )
      assertEquals(
        index.queryByInterval(
          Row("zebra"),
          Row(),
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        leafChildren.takeRight(3),
      )
    }
  }

  test("IterateFromUntil") {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("from", "idx")
      writeIndex(
        file,
        stringsWithDups,
        stringsWithDups.indices.map(i => Row()),
        TStruct.empty,
        branchingFactor,
      )
      val index = indexReader(file, TStruct.empty)

      val uniqueStrings =
        stringsWithDups.distinct ++ Array("aardvark", "crow", "elk", "otter", "zoo")
      uniqueStrings.foreach { s =>
        var start = stringsWithDups.indexWhere(s <= _)
        if (start == -1)
          start = stringsWithDups.length
        assertEquals(
          index.iterateFrom(s).toFastSeq,
          leafsWithDups.slice(
            start,
            stringsWithDups.length,
          ),
        )

        val end = stringsWithDups.lastIndexWhere(s > _) + 1
        assertEquals(index.iterateUntil(s).toFastSeq, leafsWithDups.slice(0, end))
      }
    }
  }
}
