package is.hail.utils

import is.hail.HailSuite
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.{toRichIterable, toRichOrderedArray, toRichOrderedSeq}
import is.hail.io.fs.HadoopFS
import is.hail.sparkextras.implicits._

import org.apache.spark.storage.StorageLevel
import org.scalacheck.Gen
import org.scalacheck.Gen.containerOf
import org.scalacheck.Prop.forAll

class UtilsSuite extends HailSuite with munit.ScalaCheckSuite {
  test("D_==") {
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

  test("FlushDouble") {
    assert(flushDouble(8.0e-308) == 8.0e-308)
    assert(flushDouble(-8.0e-308) == -8.0e-308)
    assert(flushDouble(8.0e-309) == 0.0)
    assert(flushDouble(-8.0e-309) == 0.0)
    assert(flushDouble(0.0) == 0.0)
  }

  test("AreDistinct") {
    assert(Array().areDistinct())
    assert(Array(1).areDistinct())
    assert(Array(1, 2).areDistinct())
    assert(!Array(1, 1).areDistinct())
    assert(!Array(1, 2, 1).areDistinct())
  }

  test("IsIncreasing") {
    assert(Seq[Int]().isIncreasing)
    assert(Seq(1).isIncreasing)
    assert(Seq(1, 2).isIncreasing)
    assert(!Seq(1, 1).isIncreasing)
    assert(!Seq(1, 2, 1).isIncreasing)

    assert(Array(1, 2).isIncreasing)
  }

  test("IsSorted") {
    assert(Seq[Int]().isSorted)
    assert(Seq(1).isSorted)
    assert(Seq(1, 2).isSorted)
    assert(Seq(1, 1).isSorted)
    assert(!Seq(1, 2, 1).isSorted)

    assert(Array(1, 1).isSorted)
  }

  test("PairRDDNoDup") {
    val answer1 =
      Array((1, (1, Option(1))), (2, (4, Option(2))), (3, (9, Option(3))), (4, (16, Option(4))))
    val pairRDD1 = sc.parallelize(ArraySeq(1, 2, 3, 4)).map(i => (i, i * i))
    val pairRDD2 = sc.parallelize(ArraySeq(1, 2, 3, 4, 1, 2, 3, 4)).map(i => (i, i))
    val join = pairRDD1.leftOuterJoin(pairRDD2.distinct())

    assert(join.collect().sortBy(t => t._1) sameElements answer1)
    assertEquals(join.count(), 4L)
  }

  test("ForallExists") {
    val rdd1 = sc.parallelize(ArraySeq(1, 2, 3, 4, 5))

    assert(rdd1.forall(_ > 0))
    assert(!rdd1.forall(_ <= 0))
    assert(!rdd1.forall(_ < 3))
    assert(rdd1.exists(_ > 4))
    assert(!rdd1.exists(_ < 0))
  }

  test("SortFileListEntry") {
    val fs = new HadoopFS(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

    val partFileNames = fs.glob(getTestResource("part-*"))
      .sortBy(fileListEntry => is.hail.io.fs.getPartNumber(fileListEntry.getPath)).map(
        _.getPath.split(
          "/"
        ).last
      )

    assert(partFileNames(0) == "part-40001")
    assert(partFileNames(1) == "part-100001")
  }

  test("storageLevelString") {
    val sls = List(
      "NONE", "DISK_ONLY", "DISK_ONLY_2", "MEMORY_ONLY", "MEMORY_ONLY_2", "MEMORY_ONLY_SER",
      "MEMORY_ONLY_SER_2",
      "MEMORY_AND_DISK", "MEMORY_AND_DISK_2", "MEMORY_AND_DISK_SER", "MEMORY_AND_DISK_SER_2",
      "OFF_HEAP")

    sls.foreach(sl => StorageLevel.fromString(sl))
  }

  test("DictionaryOrdering") {
    val stringList = Seq("Cats", "Crayon", "Dog")

    val longestToShortestLength = Ordering.by[String, Int](-_.length)
    val byFirstLetter = Ordering.by[String, Char](_.charAt(0))
    val alphabetically = Ordering.String

    val ord1 = dictionaryOrdering(alphabetically, byFirstLetter, longestToShortestLength)
    assertEquals(stringList.sorted(ord1), stringList)
    val ord2 = dictionaryOrdering(byFirstLetter, longestToShortestLength)
    assertEquals(stringList.sorted(ord2), Seq("Crayon", "Cats", "Dog"))
  }

  property("CollectAsSet") =
    forAll(containerOf[ArraySeq, Int](Gen.choose(-1000, 1000)), Gen.choose(1, 10)) {
      case (values, parts) =>
        val rdd = sc.parallelize(values, numSlices = parts)
        rdd.collectAsSet() == rdd.collect().toSet
    }

  test("DigitsNeeded") {
    assertEquals(digitsNeeded(0), 1)
    assertEquals(digitsNeeded(1), 1)
    assertEquals(digitsNeeded(7), 1)
    assertEquals(digitsNeeded(9), 1)
    assertEquals(digitsNeeded(13), 2)
    assertEquals(digitsNeeded(30173), 5)
  }

  test("toMapUniqueEmpty") {
    assertEquals(toMapIfUnique(Seq[(Int, Int)]())(x => x % 2), Right(Map[Int, Int]()))
  }

  test("toMapUniqueSingleton") {
    assertEquals(toMapIfUnique(Seq(1 -> 2))(x => x % 2), Right(Map(1 -> 2)))
  }

  test("toMapUniqueSmallNoDupe") {
    assertEquals(
      toMapIfUnique(Seq(1 -> 2, 3 -> 6, 10 -> 2))(x => x % 5),
      Right(Map(1 -> 2, 3 -> 6, 0 -> 2)),
    )
  }

  test("toMapUniqueSmallDupes") {
    assertEquals(toMapIfUnique(Seq(1 -> 2, 6 -> 6, 10 -> 2))(x => x % 5), Left(Map(1 -> Seq(1, 6))))
  }

  test("ItemPartition") {
    def check(n: Int, k: Int): Unit = {
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

    check(0, 0)
    check(0, 4)
    check(2, 4)
    check(2, 5)
    check(12, 4)
    check(12, 5)
  }

  test("TreeAggDepth") {
    assertEquals(treeAggDepth(20, 20), 1)
    assertEquals(treeAggDepth(20, 19), 2)
    assertEquals(treeAggDepth(399, 20), 2)
    assertEquals(treeAggDepth(400, 20), 2)
    assertEquals(treeAggDepth(401, 20), 3)
    assertEquals(treeAggDepth(0, 20), 1)
  }

  test("Merge") {
    val lt: (Int, Int) => Boolean =
      _ < _

    val empty: IndexedSeq[Int] =
      IndexedSeq.empty

    assertEquals(merge(empty, empty, lt), empty)

    val ones: IndexedSeq[Int] =
      ArraySeq(1)

    assertEquals(merge(ones, empty, lt), ones)
    assertEquals(merge(empty, ones, lt), ones)

    val twos: IndexedSeq[Int] =
      ArraySeq(2)

    assertEquals(merge(ones, twos, lt), (1 to 2).toIndexedSeq)
    assertEquals(merge(twos, ones, lt), (1 to 2).toIndexedSeq)

    val threes: IndexedSeq[Int] =
      ArraySeq(3)

    assertEquals(merge(twos, ones ++ threes, lt), (1 to 3).toIndexedSeq)

    // inputs need to be sorted
    assertEquals(merge(twos, threes ++ ones, lt), IndexedSeq(2, 3, 1))
  }

}
