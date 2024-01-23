package is.hail.io

import is.hail.HailSuite
import is.hail.check.Gen._
import is.hail.check.Prop._
import is.hail.check.Properties

import org.testng.annotations.Test

class IndexBTreeSuite extends HailSuite {

  object Spec extends Properties("BTree") {

    val arraySizeGenerator = for {
      depth <- frequency((4, const(1)), (5, const(2)), (1, const(3)))
      arraySize <- choose(
        math.max(1, math.pow(10, (depth - 1) * math.log10(1024)).toInt),
        math.min(1100000, math.pow(10, depth * math.log10(1024)).toInt),
      )
    } yield (depth, arraySize)

    def fillRandomArray(arraySize: Int): Array[Long] = {
      val randArray = new Array[Long](arraySize)
      var pos = 24.toLong
      var nIndices = 0
      while (nIndices != arraySize) {
        randArray(nIndices) = pos
        pos += choose(5.toLong, 5000.toLong).sample()
        nIndices += 1
      }
      randArray
    }

    property("query gives same answer as array") =
      forAll(arraySizeGenerator) { case (depth: Int, arraySize: Int) =>
        val arrayRandomStarts = fillRandomArray(arraySize)
        val maxLong = arrayRandomStarts.takeRight(1)(0)
        val index = ctx.createTmpPath("testBtree", "idx")

        fs.delete(index, true)
        IndexBTree.write(arrayRandomStarts, index, fs)
        val btree = new IndexBTree(index, fs)

        val indexSize = fs.getFileSize(index)
        val padding = 1024 - (arraySize % 1024)
        val numEntries = arraySize + padding + (1 until depth).map {
          math.pow(1024, _).toInt
        }.sum

        // make sure index size is correct
        val indexCorrectSize = if (indexSize == (numEntries * 8)) true else false

        // make sure depth is correct
        val estimatedDepth = btree.calcDepth()
        val depthCorrect = if (estimatedDepth == depth) true else false

        // make sure query is correct
        val queryCorrect = if (arrayRandomStarts.length < 100)
          arrayRandomStarts.forall { case (l) => btree.queryIndex(l - 1).contains(l) }
        else {
          val randomIndices = Array(0) ++ Array.fill(100)(choose(0, arraySize - 1).sample())
          randomIndices.map(arrayRandomStarts).forall { case (l) =>
            btree.queryIndex(l - 1).contains(l)
          }
        }

        if (!depthCorrect || !indexCorrectSize || !queryCorrect)
          println(s"depth=$depthCorrect indexCorrect=$indexCorrectSize queryCorrect=$queryCorrect")

        btree.close()
        depthCorrect && indexCorrectSize && queryCorrect
      }
  }

  @Test def test(): Unit =
    Spec.check()

  @Test def oneVariant(): Unit = {
    val index = Array(24.toLong)
    val fileSize = 30 // made-up value greater than index
    val idxFile = ctx.createTmpPath("testBtree_1variant", "idx")

    fs.delete(idxFile, recursive = true)
    IndexBTree.write(index, idxFile, fs)
    val btree = new IndexBTree(idxFile, fs)

    intercept[IllegalArgumentException] {
      btree.queryIndex(-5)
    }

    assert(btree.queryIndex(0).contains(24))
    assert(btree.queryIndex(10).contains(24))
    assert(btree.queryIndex(20).contains(24))
    assert(btree.queryIndex(24).contains(24))
    assert(btree.queryIndex(25).isEmpty)
    assert(btree.queryIndex(fileSize - 1).isEmpty)
  }

  @Test def zeroVariants(): Unit = {
    intercept[IllegalArgumentException] {
      val index = Array[Long]()
      val idxFile = ctx.createTmpPath("testBtree_0variant", "idx")
      fs.delete(idxFile, recursive = true)
      IndexBTree.write(index, idxFile, fs)
    }
  }

  @Test def testMultipleOfBranchingFactorDoesNotAddUnnecessaryElements(): Unit = {
    val in = Array[Long](10, 9, 8, 7, 6, 5, 4, 3)
    val bigEndianBytes = Array[Byte](
      0, 0, 0, 0, 0, 0, 0, 10,
      0, 0, 0, 0, 0, 0, 0, 9,
      0, 0, 0, 0, 0, 0, 0, 8,
      0, 0, 0, 0, 0, 0, 0, 7,
      0, 0, 0, 0, 0, 0, 0, 6,
      0, 0, 0, 0, 0, 0, 0, 5,
      0, 0, 0, 0, 0, 0, 0, 4,
      0, 0, 0, 0, 0, 0, 0, 3)
    assert(IndexBTree.btreeBytes(in, branchingFactor = 8)
      sameElements bigEndianBytes)
  }

  @Test def writeReadMultipleOfBranchingFactorDoesNotError(): Unit = {
    val idxFile = ctx.createTmpPath("btree")
    IndexBTree.write(
      Array.tabulate(1024)(i => i),
      idxFile,
      fs,
    )
    val index = new IndexBTree(idxFile, fs)
    assert(index.queryIndex(33).contains(33L))
  }

  @Test def queryArrayPositionAndFileOffsetIsCorrectSmallArray(): Unit = {
    val f = ctx.createTmpPath("btree")
    val v = Array[Long](1, 2, 3, 40, 50, 60, 70)
    val branchingFactor = 1024
    IndexBTree.write(v, f, fs, branchingFactor = branchingFactor)
    val bt = new IndexBTree(f, fs, branchingFactor = branchingFactor)
    assert(bt.queryArrayPositionAndFileOffset(1).contains((0, 1)))
    assert(bt.queryArrayPositionAndFileOffset(2).contains((1, 2)))
    assert(bt.queryArrayPositionAndFileOffset(3).contains((2, 3)))
    for (i <- 4 to 40)
      assert(bt.queryArrayPositionAndFileOffset(i).contains((3, 40)), s"$i")
    for (i <- 41 to 50)
      assert(bt.queryArrayPositionAndFileOffset(i).contains((4, 50)), s"$i")
    assert(bt.queryArrayPositionAndFileOffset(65).contains((6, 70)))
    assert(bt.queryArrayPositionAndFileOffset(70).contains((6, 70)))
    assert(bt.queryArrayPositionAndFileOffset(71).isEmpty)
  }

  @Test def queryArrayPositionAndFileOffsetIsCorrectTwoLevelsArray(): Unit = {
    def sqr(x: Long) = x * x
    val f = ctx.createTmpPath("btree")
    val v = Array.tabulate(1025)(x => sqr(x))
    val branchingFactor = 1024
    IndexBTree.write(v, f, fs, branchingFactor = branchingFactor)
    val bt = new IndexBTree(f, fs, branchingFactor = branchingFactor)
    assert(bt.queryArrayPositionAndFileOffset(sqr(1022)).contains((1022, sqr(1022))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1022) + 1).contains((1023, sqr(1023))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1023) - 1).contains((1023, sqr(1023))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1023)).contains((1023, sqr(1023))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1023) + 1).contains((1024, sqr(1024))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024) - 1).contains((1024, sqr(1024))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024)).contains((1024, sqr(1024))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1024) + 1).isEmpty)

    assert(bt.queryArrayPositionAndFileOffset(0).contains((0, sqr(0))))
    assert(bt.queryArrayPositionAndFileOffset(1).contains((1, sqr(1))))
    assert(bt.queryArrayPositionAndFileOffset(2).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(3).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(4).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(5).contains((3, sqr(3))))
  }

  @Test def queryArrayPositionAndFileOffsetIsCorrectThreeLevelsArray(): Unit = {
    def sqr(x: Long) = x * x
    val f = ctx.createTmpPath("btree")
    val v = Array.tabulate(1024 * 1024 + 1)(x => sqr(x))
    val branchingFactor = 1024
    IndexBTree.write(v, f, fs, branchingFactor = branchingFactor)
    val bt = new IndexBTree(f, fs, branchingFactor = branchingFactor)
    assert(bt.queryArrayPositionAndFileOffset(sqr(1022)).contains((1022, sqr(1022))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1022) + 1).contains((1023, sqr(1023))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1023) - 1).contains((1023, sqr(1023))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1023)).contains((1023, sqr(1023))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1023) + 1).contains((1024, sqr(1024))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024) - 1).contains((1024, sqr(1024))))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024)).contains((1024, sqr(1024))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1024) + 1).contains((1025, sqr(1025))))

    assert(bt.queryArrayPositionAndFileOffset(0).contains((0, sqr(0))))
    assert(bt.queryArrayPositionAndFileOffset(1).contains((1, sqr(1))))
    assert(bt.queryArrayPositionAndFileOffset(2).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(3).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(4).contains((2, sqr(2))))
    assert(bt.queryArrayPositionAndFileOffset(5).contains((3, sqr(3))))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1024 * 1024 - 1)).contains((
      1024 * 1024 - 1,
      sqr(1024 * 1024 - 1),
    )))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024 * 1024 - 1) + 1).contains((
      1024 * 1024,
      sqr(1024 * 1024),
    )))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1024 * 1024)).contains((
      1024 * 1024,
      sqr(1024 * 1024),
    )))
    assert(bt.queryArrayPositionAndFileOffset(sqr(1024 * 1024) - 1).contains((
      1024 * 1024,
      sqr(1024 * 1024),
    )))

    assert(bt.queryArrayPositionAndFileOffset(sqr(1024 * 1024) + 1).isEmpty)
  }

  @Test def onDiskBTreeIndexToValueSmallCorrect(): Unit = {
    val f = ctx.createTmpPath("btree")
    val v = Array[Long](1, 2, 3, 4, 5, 6, 7)
    val branchingFactor = 3
    try {
      IndexBTree.write(v, f, fs, branchingFactor)
      val bt = new OnDiskBTreeIndexToValue(f, fs, branchingFactor)
      assert(bt.positionOfVariants(Array()) sameElements Array[Long]())
      assert(bt.positionOfVariants(Array(5)) sameElements Array(6L))

      val indices = Seq(0, 5, 1, 6)
      val actual = bt.positionOfVariants(indices.toArray)
      val expected = indices.sorted.map(v)
      assert(
        actual sameElements expected,
        s"${actual.toSeq} not same elements as expected ${expected.toSeq}",
      )
    } catch {
      case t: Throwable =>
        throw new RuntimeException(
          "exception while checking BTree: " + IndexBTree.toString(v, branchingFactor),
          t,
        )
    }
  }

  @Test def onDiskBTreeIndexToValueRandomized(): Unit = {
    val g = for {
      longs <- buildableOf[Array](choose(0L, Long.MaxValue))
      indices <- buildableOf[Array](choose(0, longs.length - 1))
      branchingFactor <- choose(2, 1024)
    } yield (indices, longs, branchingFactor)
    forAll(g) { case (indices, longs, branchingFactor) =>
      val f = ctx.createTmpPath("test")
      try {
        IndexBTree.write(longs, f, fs, branchingFactor)
        val bt = new OnDiskBTreeIndexToValue(f, fs, branchingFactor)
        val actual = bt.positionOfVariants(indices.toArray)
        val expected = indices.sorted.map(longs)
        assert(
          actual sameElements expected,
          s"${actual.toSeq} not same elements as expected ${expected.toSeq}",
        )
      } catch {
        case t: Throwable =>
          throw new RuntimeException(
            "exception while checking BTree: " + IndexBTree.toString(longs, branchingFactor),
            t,
          )
      }
      true
    }.check()
  }

  @Test def onDiskBTreeIndexToValueFourLayers(): Unit = {
    val longs = Array.tabulate(3 * 3 * 3 * 3)(x => x.toLong)
    val indices = Array(0, 3, 10, 20, 26, 27, 34, 55, 79, 80)
    val f = ctx.createTmpPath("btree")
    val branchingFactor = 3
    try {
      IndexBTree.write(longs, f, fs, branchingFactor)
      val bt = new OnDiskBTreeIndexToValue(f, fs, branchingFactor)
      val expected = indices.sorted.map(longs)
      val actual = bt.positionOfVariants(indices.toArray)
      assert(
        actual sameElements expected,
        s"${actual.toSeq} not same elements as expected ${expected.toSeq}",
      )
    } catch {
      case t: Throwable =>
        throw new RuntimeException(
          "exception while checking BTree: " + IndexBTree.toString(longs, branchingFactor),
          t,
        )
    }
  }

  @Test def calcDepthIsCorrect(): Unit = {
    def sqr(x: Long) = x * x
    def cube(x: Long) = x * x * x

    def f(x: Long) = IndexBTree.calcDepth(x, 1024)

    assert(f(1) == 1)
    assert(f(1023) == 1)
    assert(f(1024) == 1)
    assert(f(1025) == 2)

    assert(f(sqr(1024) - 1) == 2)
    assert(f(sqr(1024)) == 2)
    assert(f(sqr(1024) + 1024) == 2)
    assert(f(sqr(1024) + 1024 + 1) == 3)

    assert(f(cube(1024) - 1) == 3)
    assert(f(cube(1024)) == 3)
    assert(f(cube(1024) + sqr(1024)) == 3)
    assert(f(cube(1024) + sqr(1024) + 1024) == 3)
    assert(f(cube(1024) + sqr(1024) + 1024 + 1) == 4)
  }
}
