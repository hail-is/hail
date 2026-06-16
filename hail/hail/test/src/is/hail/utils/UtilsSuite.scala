package is.hail.utils

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichIterable, toRichOrderedArray, toRichOrderedSeq}
import is.hail.io.fs.HadoopFS
import is.hail.sparkextras.implicits._

import org.apache.spark.storage.StorageLevel
import org.junit.jupiter.api.Test
import org.scalacheck.Gen
import org.scalacheck.Gen.containerOf
import org.scalacheck.Prop.forAll

class UtilsSuite {
  @Test def testD_==(): Unit = {
    assert(D_==(1, 1))
    assert(D_==(1, 1 + 1e-7))
    assert(!D_==(1, 1 + 1e-5))
    assert(D_==(1e10, 1e10 + 1))
    assert(!D_==(1e-10, 2e-10))
    assert(D_==(0.0, 0.0))
    assert(D_!=(1e-307, 0.0))
    assert(D_==(1e-308, 0.0))
    assert(D_==(1e-320, -1e-320))
  }

  @Test def testFlushDouble(): Unit = {
    assertEq(flushDouble(8.0e-308), 8.0e-308)
    assertEq(flushDouble(-8.0e-308), -8.0e-308)
    assertEq(flushDouble(8.0e-309), 0.0)
    assertEq(flushDouble(-8.0e-309), 0.0)
    assertEq(flushDouble(0.0), 0.0)
  }

  @Test def testAreDistinct(): Unit = {
    assert(Array().areDistinct())
    assert(Array(1).areDistinct())
    assert(Array(1, 2).areDistinct())
    assert(!Array(1, 1).areDistinct())
    assert(!Array(1, 2, 1).areDistinct())
  }

  @Test def testIsIncreasing(): Unit = {
    assert(Seq[Int]().isIncreasing)
    assert(Seq(1).isIncreasing)
    assert(Seq(1, 2).isIncreasing)
    assert(!Seq(1, 1).isIncreasing)
    assert(!Seq(1, 2, 1).isIncreasing)

    assert(Array(1, 2).isIncreasing)
  }

  @Test def testIsSorted(): Unit = {
    assert(Seq[Int]().isSorted)
    assert(Seq(1).isSorted)
    assert(Seq(1, 2).isSorted)
    assert(Seq(1, 1).isSorted)
    assert(!Seq(1, 2, 1).isSorted)

    assert(Array(1, 1).isSorted)
  }

  @Test def testPairRDDNoDup(implicit ctx: ExecuteContext): Unit = {
    val sc = ctx.backend.asSpark.sc
    val answer1 =
      Array((1, (1, Option(1))), (2, (4, Option(2))), (3, (9, Option(3))), (4, (16, Option(4))))
    val pairRDD1 = sc.parallelize(ArraySeq(1, 2, 3, 4)).map(i => (i, i * i))
    val pairRDD2 = sc.parallelize(ArraySeq(1, 2, 3, 4, 1, 2, 3, 4)).map(i => (i, i))
    val join = pairRDD1.leftOuterJoin(pairRDD2.distinct())

    assert(join.collect().sortBy(t => t._1) sameElements answer1)
    assertEq(join.count(), 4L)
  }

  @Test def testForallExists(implicit ctx: ExecuteContext): Unit = {
    val sc = ctx.backend.asSpark.sc
    val rdd1 = sc.parallelize(ArraySeq(1, 2, 3, 4, 5))

    assert(rdd1.forall(_ > 0))
    assert(!rdd1.forall(_ <= 0))
    assert(!rdd1.forall(_ < 3))
    assert(rdd1.exists(_ > 4))
    assert(!rdd1.exists(_ < 0))
  }

  @Test def testSortFileListEntry(implicit ctx: ExecuteContext): Unit = {
    val sc = ctx.backend.asSpark.sc
    val fs = new HadoopFS(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

    val partFileNames = fs.glob(getTestResource("part-*"))
      .sortBy(fileListEntry => is.hail.io.fs.getPartNumber(fileListEntry.getPath)).map(
        _.getPath.split(
          "/"
        ).last
      )

    assert(partFileNames(0) == "part-40001" && partFileNames(1) == "part-100001")
  }

  @Test def storageLevelStringTest(): Unit = {
    val sls = List(
      "NONE", "DISK_ONLY", "DISK_ONLY_2", "MEMORY_ONLY", "MEMORY_ONLY_2", "MEMORY_ONLY_SER",
      "MEMORY_ONLY_SER_2",
      "MEMORY_AND_DISK", "MEMORY_AND_DISK_2", "MEMORY_AND_DISK_SER", "MEMORY_AND_DISK_SER_2",
      "OFF_HEAP")

    sls.foreach(sl => StorageLevel.fromString(sl))
  }

  @Test def testDictionaryOrdering(): Unit = {
    val stringList = Seq("Cats", "Crayon", "Dog")

    val longestToShortestLength = Ordering.by[String, Int](-_.length)
    val byFirstLetter = Ordering.by[String, Char](_.charAt(0))
    val alphabetically = Ordering.String

    val ord1 = dictionaryOrdering(alphabetically, byFirstLetter, longestToShortestLength)
    assertEq(stringList.sorted(ord1), stringList)
    val ord2 = dictionaryOrdering(byFirstLetter, longestToShortestLength)
    assertEq(stringList.sorted(ord2), Seq("Crayon", "Cats", "Dog"))
  }

  @Test def testCollectAsSet(implicit ctx: ExecuteContext): Unit = {
    val sc = ctx.backend.asSpark.sc
    check(
      forAll(containerOf[ArraySeq, Int](Gen.choose(-1000, 1000)), Gen.choose(1, 10)) {
        (values, parts) =>
          val rdd = sc.parallelize(values, numSlices = parts)
          assertEq(rdd.collectAsSet(), rdd.collect().toSet)
      }
    )
  }

  @Test def testDigitsNeeded(): Unit = {
    assertEq(digitsNeeded(0), 1)
    assertEq(digitsNeeded(1), 1)
    assertEq(digitsNeeded(7), 1)
    assertEq(digitsNeeded(9), 1)
    assertEq(digitsNeeded(13), 2)
    assertEq(digitsNeeded(30173), 5)
  }

  @Test def toMapUniqueEmpty(): Unit =
    assertEq(toMapIfUnique(Seq[(Int, Int)]())(x => x % 2), Right(Map()))

  @Test def toMapUniqueSingleton(): Unit =
    assertEq(toMapIfUnique(Seq(1 -> 2))(x => x % 2), Right(Map(1 -> 2)))

  @Test def toMapUniqueSmallNoDupe(): Unit =
    assertEq(
      toMapIfUnique(Seq(1 -> 2, 3 -> 6, 10 -> 2))(x => x % 5),
      Right(Map(1 -> 2, 3 -> 6, 0 -> 2)),
    )

  @Test def toMapUniqueSmallDupes(): Unit =
    assertEq(toMapIfUnique(Seq(1 -> 2, 6 -> 6, 10 -> 2))(x => x % 5), Left(Map(1 -> Seq(1, 6))))

  @Test def testItemPartition(): Unit = {
    def test(n: Int, k: Int): Unit = {
      val a = new Array[Int](k)
      var prevj = 0
      for (i <- 0 until n) {
        val j = itemPartition(i, n, k)

        assert(j >= 0)
        assert(j < k)
        a(j) += 1

        assert(prevj <= j)
        prevj = j
      }
      val p = partition(n, k)
      assert(a sameElements p)
    }

    test(0, 0)
    test(0, 4)
    test(2, 4)
    test(2, 5)
    test(12, 4)
    test(12, 5)
  }

  @Test def testTreeAggDepth(): Unit = {
    assertEq(treeAggDepth(20, 20), 1)
    assertEq(treeAggDepth(20, 19), 2)
    assertEq(treeAggDepth(399, 20), 2)
    assertEq(treeAggDepth(400, 20), 2)
    assertEq(treeAggDepth(401, 20), 3)
    assertEq(treeAggDepth(0, 20), 1)
  }

  @Test def testMerge(): Unit = {
    val lt: (Int, Int) => Boolean =
      _ < _

    val empty: IndexedSeq[Int] =
      IndexedSeq.empty

    assertEq(merge(empty, empty, lt), empty)

    val ones: IndexedSeq[Int] =
      ArraySeq(1)

    assertEq(merge(ones, empty, lt), ones)
    assertEq(merge(empty, ones, lt), ones)

    val twos: IndexedSeq[Int] =
      ArraySeq(2)

    assertEq(merge(ones, twos, lt), (1 to 2))
    assertEq(merge(twos, ones, lt), (1 to 2))

    val threes: IndexedSeq[Int] =
      ArraySeq(3)

    assertEq(merge(twos, ones ++ threes, lt), (1 to 3))

    // inputs need to be sorted
    assertEq(merge(twos, threes ++ ones, lt), Seq(2, 3, 1))
  }

}
