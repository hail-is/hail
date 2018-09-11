package is.hail.utils

import is.hail.SparkSuite
import is.hail.check.Arbitrary._
import is.hail.check.{Gen, Prop}
import is.hail.utils.richUtils.RichHadoopConfiguration
import is.hail.variant._
import org.apache.spark.storage.StorageLevel
import org.testng.annotations.Test

class UtilsSuite extends SparkSuite {
  @Test def testD_==() {
    assert(D_==(1, 1))
    assert(D_==(1, 1 + 1E-7))
    assert(!D_==(1, 1 + 1E-5))
    assert(D_==(1E10, 1E10 + 1))
    assert(!D_==(1E-10, 2E-10))
    assert(D_==(0.0, 0.0))
    assert(D_!=(1E-307, 0.0))
    assert(D_==(1E-308, 0.0))
    assert(D_==(1E-320, -1E-320))
  }

  @Test def testFlushDouble() {
    assert(flushDouble(8.0E-308) == 8.0E-308)
    assert(flushDouble(-8.0E-308) == -8.0E-308)
    assert(flushDouble(8.0E-309) == 0.0)
    assert(flushDouble(-8.0E-309) == 0.0)
    assert(flushDouble(0.0) == 0.0)
  }

  @Test def testAreDistinct() {
    assert(Array().areDistinct())
    assert(Array(1).areDistinct())
    assert(Array(1, 2).areDistinct())
    assert(!Array(1, 1).areDistinct())
    assert(!Array(1, 2, 1).areDistinct())
  }

  @Test def testIsIncreasing() {
    assert(Seq[Int]().isIncreasing)
    assert(Seq(1).isIncreasing)
    assert(Seq(1, 2).isIncreasing)
    assert(!Seq(1, 1).isIncreasing)
    assert(!Seq(1, 2, 1).isIncreasing)

    assert(Array(1, 2).isIncreasing)
  }

  @Test def testIsSorted() {
    assert(Seq[Int]().isSorted)
    assert(Seq(1).isSorted)
    assert(Seq(1, 2).isSorted)
    assert(Seq(1, 1).isSorted)
    assert(!Seq(1, 2, 1).isSorted)

    assert(Array(1, 1).isSorted)
  }

  @Test def testHadoopStripCodec() {
    assert(hadoopConf.stripCodec("file.tsv") == "file.tsv")
    assert(hadoopConf.stripCodec("file.tsv.gz") == "file.tsv")
    assert(hadoopConf.stripCodec("file.tsv.bgz") == "file.tsv")
    assert(hadoopConf.stripCodec("file.tsv.lz4") == "file.tsv")
    assert(hadoopConf.stripCodec("file") == "file")
  }

  @Test def testPairRDDNoDup() {
    val answer1 = Array((1, (1, Option(1))), (2, (4, Option(2))), (3, (9, Option(3))), (4, (16, Option(4))))
    val pairRDD1 = sc.parallelize(Array(1, 2, 3, 4)).map { i => (i, i * i) }
    val pairRDD2 = sc.parallelize(Array(1, 2, 3, 4, 1, 2, 3, 4)).map { i => (i, i) }
    val join = pairRDD1.leftOuterJoin(pairRDD2.distinct)

    assert(join.collect().sortBy(t => t._1) sameElements answer1)
    assert(join.count() == 4)
  }

  @Test def testForallExists() {
    val rdd1 = sc.parallelize(Array(1, 2, 3, 4, 5))

    assert(rdd1.forall(_ > 0))
    assert(!rdd1.forall(_ <= 0))
    assert(!rdd1.forall(_ < 3))
    assert(rdd1.exists(_ > 4))
    assert(!rdd1.exists(_ < 0))
  }

  @Test def testSortFileStatus() {
    val rhc = new RichHadoopConfiguration(sc.hadoopConfiguration)

    val partFileNames = rhc.glob("src/test/resources/part-*").sortBy(fs => getPartNumber(fs.getPath.getName)).map(_.getPath.getName)

    assert(partFileNames(0) == "part-40001" && partFileNames(1) == "part-100001")
  }

  @Test def storageLevelStringTest() = {
    val sls = List(
      "NONE", "DISK_ONLY", "DISK_ONLY_2", "MEMORY_ONLY", "MEMORY_ONLY_2", "MEMORY_ONLY_SER", "MEMORY_ONLY_SER_2",
      "MEMORY_AND_DISK", "MEMORY_AND_DISK_2", "MEMORY_AND_DISK_SER", "MEMORY_AND_DISK_SER_2", "OFF_HEAP")

    sls.foreach { sl => StorageLevel.fromString(sl) }
  }

  @Test def testDictionaryOrdering() {
    val stringList = Seq("Cats", "Crayon", "Dog")

    val longestToShortestLength = Ordering.by[String, Int](-_.length)
    val byFirstLetter = Ordering.by[String, Char](_.charAt(0))
    val alphabetically = Ordering.String

    val ord1 = dictionaryOrdering(alphabetically, byFirstLetter, longestToShortestLength)
    assert(stringList.sorted(ord1) == stringList)
    val ord2 = dictionaryOrdering(byFirstLetter, longestToShortestLength)
    assert(stringList.sorted(ord2) == Seq("Crayon", "Cats", "Dog"))
  }

  @Test def testUInt() {
    assert(UInt((1L << 32) - 1) == 4294967295L)
    assert(UInt(4294967295L) == 4294967295L)
    assert(UInt(327886) == 327886)
    assert(UInt(4294967295L) == 4294967295d)

    assert(UInt(2147483647) + UInt(5) == UInt(2147483652L))
    assert(UInt(2147483647) + 5 == UInt(2147483652L))
    assert(UInt(2147483647) + 0.5 == 2147483647.5)
    assert(UInt(2147483647) + 5L == 2147483652L)

    assert(UInt(2147483647) - UInt(5) == UInt(2147483642L))
    assert(UInt(2147483647) - 1 == UInt(2147483646L))
    assert(UInt(2147483647) - 0.5 == 2147483646.5)
    assert(UInt(2147483647) - 1L == 2147483646L)

    assert(UInt(2147483647) * UInt(2) == UInt(4294967294L))
    assert(UInt(2147483647) * 2 == UInt(4294967294L))
    assert(UInt(2147483647) * 1.2 == 2.5769803764E9)
    assert(UInt(2147483647) * 2L == 4294967294L)

    assert(UInt(2147483647) / UInt(2) == UInt(1073741823L))
    assert(UInt(2147483647) / 2 == UInt(1073741823L))
    assert(UInt(2147483647) / 2.0 == 1073741823.5)
    assert(UInt(2147483647) / 2L == 1073741823L)

    assert(UInt(2147483647) == UInt(2147483647))
    assert(UInt(2147483647) == 2147483647)
    assert(UInt(2147483647) == 2147483647d)
    assert(UInt(2147483647) == 2147483647L)

    assert(UInt(4294967295L) != UInt(0))
    assert(UInt(4294967295L) != -1)
    assert(UInt(4294967295L) != 0.5)
    assert(UInt(4294967295L) != -1L)

    assert(UInt(4294967295L) > UInt(0))
    assert(UInt(4294967295L) > -1)
    assert(UInt(5) > 4.5)
    assert(UInt(5) > -1L)

    assert(UInt(4294967295L) >= UInt(0))
    assert(UInt(5) >= -1)
    assert(UInt(5) >= 4.5)
    assert(UInt(5) >= -1L)

    assert(UInt(0) < UInt(4294967295L))
    assert(UInt(5000L) < 5500)
    assert(UInt(4294967295L) < 4294967299.5)
    assert(UInt(4294967295L) < 4294967299L)

    assert(UInt(0) <= UInt(4294967295L))
    assert(UInt(5000L) <= 5500)
    assert(UInt(4294967295L) <= 4294967299.5)
    assert(UInt(4294967295L) <= 4294967299L)

    assert(UInt(4294967295L) == (UInt(4294967295L) + UInt(0)))

    intercept[AssertionError](UInt(-5).toInt)
    intercept[AssertionError](UInt(5L) - UInt(4294967295L))
    intercept[AssertionError](UInt(4294967295L * 10))
    intercept[AssertionError](UInt(4294967294L) + UInt(10))
    intercept[AssertionError](UInt(-1) * UInt(2))
    intercept[AssertionError](UInt(-3) + UInt(5) == UInt(2))
  }

  @Test def testCollectAsSet() {
    Prop.forAll(Gen.buildableOf[Array](Gen.choose(-1000, 1000)), Gen.choose(1, 10)) { case (values, parts) =>
      val rdd = sc.parallelize(values, numSlices = parts)
      rdd.collectAsSet() == rdd.collect().toSet
    }.check()
  }

  @Test def testDigitsNeeded() {
    assert(digitsNeeded(0) == 1)
    assert(digitsNeeded(1) == 1)
    assert(digitsNeeded(7) == 1)
    assert(digitsNeeded(9) == 1)
    assert(digitsNeeded(13) == 2)
    assert(digitsNeeded(30173) == 5)
  }

  @Test def testMangle() {
    val c1 = Array("a", "b", "c", "a", "a", "c", "a")
    val (c2, diff) = mangle(c1)
    assert(c2.toSeq == Seq("a", "b", "c", "a_1", "a_2", "c_1", "a_3"))
    assert(diff.toSeq == Seq("a" -> "a_1", "a" -> "a_2", "c" -> "c_1", "a" -> "a_3"))

    val c3 = Array("a", "b", "c", "a", "a", "c", "a")
    val (c4, diff2) = mangle(c1, "D" * _)
    assert(c4.toSeq == Seq("a", "b", "c", "aD", "aDD", "cD", "aDDD"))
    assert(diff2.toSeq == Seq("a" -> "aD", "a" -> "aDD", "c" -> "cD", "a" -> "aDDD"))
  }

  @Test def toMapUniqueEmpty() {
    assert(toMapIfUnique(Seq[(Int, Int)]())(x => x % 2) == Right(Map()))
  }

  @Test def toMapUniqueSingleton() {
    assert(toMapIfUnique(Seq(1 -> 2))(x => x % 2) == Right(Map(1 -> 2)))
  }

  @Test def toMapUniqueSmallNoDupe() {
    assert(toMapIfUnique(Seq(1 -> 2, 3 -> 6, 10 -> 2))(x => x % 5) ==
      Right(Map(1 -> 2, 3 -> 6, 0 -> 2)))
  }

  @Test def toMapUniqueSmallDupes() {
    assert(toMapIfUnique(Seq(1 -> 2, 6 -> 6, 10 -> 2))(x => x % 5) ==
      Left(Map(1 -> Seq(1, 6))))
  }
}
