package is.hail.io

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.annotations.Annotation
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterator
import is.hail.io.index._
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PInt32, PType}
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class IndexSuite {
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

  def writeReadGivesSameAsInput() = (1 to strings.length).map(i => strings.take(i))

  def writeIndex(
    file: String,
    data: IndexedSeq[Any],
    annotations: IndexedSeq[Annotation],
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int,
    attributes: Map[String, Any],
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val iw = IndexWriter.builder(ctx, keyType, annotationType, branchingFactor, attributes)(
      file,
      ctx.theHailClassLoader,
      ctx.taskContext,
      ctx.r.pool,
    )
    data.zip(annotations).zipWithIndex.foreach { case ((s, a), offset) =>
      iw.appendRow(s, offset.toLong, a)
    }
    iw.close()
  }

  def indexReader(
    file: String,
    annotationType: Type,
    keyPType: PType = PCanonicalString(),
  )(implicit ctx: ExecuteContext
  ): IndexReader =
    indexReader(file, PType.canonical(annotationType), keyPType)

  def indexReader(
    file: String,
    annotationPType: PType,
    keyPType: PType,
  )(implicit ctx: ExecuteContext
  ): IndexReader = {
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
    ).apply(ctx.theHailClassLoader, ctx.fs, file, 8, ctx.r.pool)
  }

  def writeIndex(
    file: String,
    data: IndexedSeq[String],
    annotations: IndexedSeq[Annotation],
    annotationType: Type,
    branchingFactor: Int = 2,
    attributes: Map[String, Any] = Map.empty[String, Any],
  )(implicit ctx: ExecuteContext
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

  @ParameterizedTest
  def writeReadGivesSameAsInput(data: IndexedSeq[String])(implicit ctx: ExecuteContext): Unit = {
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
      assert(ctx.fs.getFileSize(file) != 0)

      val index = indexReader(file, TStruct("a" -> TBoolean))

      assertEq(index.attributes, attributes)

      data.zipWithIndex.foreach { case (s, i) =>
        val result = index.queryByIndex(i.toLong)
        assertEq(result.key, s)
        assertEq(result.annotation, a(i))
      }

      index.close()
    }
  }

  @Test def testEmptyKeys(implicit ctx: ExecuteContext): Unit = {
    val file = ctx.createTmpPath("empty", "idx")
    writeIndex(file, ArraySeq.empty, ArraySeq.empty, TStruct("a" -> TBoolean), 2)
    assert(ctx.fs.getFileSize(file) != 0)
    val index = indexReader(file, TStruct("a" -> TBoolean))
    intercept[IllegalArgumentException](index.queryByIndex(0L)): Unit
    assert(index.queryByKey("moo").isEmpty)
    assert(index.queryByInterval("bear", "cat", includesStart = true, includesEnd = true).isEmpty)
    index.close()
  }

  @Test def testLowerBound(implicit ctx: ExecuteContext): Unit = {
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
        assertEq(index.lowerBound(s), expectedIdx.toLong)
      }
    }
  }

  @Test def testUpperBound(implicit ctx: ExecuteContext): Unit = {
    for (branchingFactor <- 2 to 5) {
      val file = ctx.createTmpPath("upperBound", "idx")
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
        assertEq(index.upperBound(s), expectedIdx.toLong)
      }
    }
  }

  @Test def testRangeIterator(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testQueryByKey(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testIntervalIterator(implicit ctx: ExecuteContext): Unit = {
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
      assertEq(
        index.queryByInterval(
          "bear",
          "bear",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 2).toFastSeq,
      )

      assertEq(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(0, 2).toFastSeq,
      )
      assertEq(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 9).toFastSeq,
      )
      assertEq(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = false,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(2, 9).toFastSeq,
      )
      assertEq(
        index.queryByInterval(
          "bear",
          "cat",
          includesStart = false,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(2, 2).toFastSeq,
      )

      // intervals with endpoint(s) not in list
      assertEq(
        index.queryByInterval(
          "cat",
          "snail",
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        index.iterator(2, 15).toFastSeq,
      )
      assertEq(
        index.queryByInterval(
          "cat",
          "snail",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(2, 15).toFastSeq,
      )
      assertEq(
        index.queryByInterval(
          "aardvark",
          "cat",
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        index.iterator(0, 9).toFastSeq,
      )
      assertEq(
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

              assertEq(
                index.queryByInterval(
                  bounds(0),
                  bounds(1),
                  includesStart,
                  includesEnd,
                ).toFastSeq,
                index.iterator(lowerBoundIdx.toLong, upperBoundIdx.toLong).toFastSeq,
              )

              if (includesStart)
                assertEq(
                  index.iterateFrom(bounds(0)).toFastSeq,
                  leafsWithDups.slice(lowerBoundIdx, stringsWithDups.length),
                )
              if (!includesEnd)
                assertEq(
                  index.iterateUntil(bounds(1)).toFastSeq,
                  leafsWithDups.slice(0, upperBoundIdx),
                )
            } else
              intercept[IllegalArgumentException](index.queryByInterval(
                bounds(0),
                bounds(1),
                includesStart,
                includesEnd,
              )): Unit
          }
        }
      }
    }
  }

  @Test def testIntervalIteratorWorksWithGeneralEndpoints(implicit ctx: ExecuteContext): Unit = {
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
      assertEq(
        index.queryByInterval(
          Row("cat", 3),
          Row("cat", 5),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.slice(3, 5),
      )
      assertEq(
        index.queryByInterval(
          Row("cat"),
          Row("cat", 5),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.slice(2, 5),
      )
      assertEq(
        index.queryByInterval(
          Row(),
          Row(),
          includesStart = true,
          includesEnd = true,
        ).toFastSeq,
        leafChildren,
      )
      assertEq(
        index.queryByInterval(
          Row(),
          Row("cat"),
          includesStart = true,
          includesEnd = false,
        ).toFastSeq,
        leafChildren.take(2),
      )
      assertEq(
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

  @Test def testIterateFromUntil(implicit ctx: ExecuteContext): Unit = {
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
        assertEq(
          index.iterateFrom(s).toFastSeq,
          leafsWithDups.slice(
            start,
            stringsWithDups.length,
          ),
        )

        val end = stringsWithDups.lastIndexWhere(s > _) + 1
        assertEq(index.iterateUntil(s).toFastSeq, leafsWithDups.slice(0, end))
      }
    }
  }
}
