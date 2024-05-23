package is.hail.linalg

import is.hail.check.{Gen, Prop}
import is.hail.check.Arbitrary.arbitrary

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowPartitionerSuite extends TestNGSuite {
  @Test
  def testGetPartition(): Unit = {
    val partitionStarts = Array[Long](0, 0, 0, 4, 5, 5, 8, 10, 10)
    val partitionCounts = Array(0, 0, 4, 1, 0, 3, 2, 0)
    val keyPart = partitionCounts.zipWithIndex.flatMap { case (count, pi) => Array.fill(count)(pi) }

    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == 8)
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }

  @Test def testFindInterval(): Unit = {
    def naiveFindInterval(a: Array[Long], key: Long): Int = {
      if (a.length == 0 || key < a(0))
        -1
      else if (key >= a(a.length - 1))
        a.length - 1
      else {
        var j = 0
        while (!(a(j) <= key && key < a(j + 1)))
          j += 1
        j
      }
    }

    val moreKeys = Array(Long.MinValue, -1000L, -1L, 0L, 1L, 1000L, Long.MaxValue)

    val g = for {
      a0 <- Gen.buildableOf[Array](arbitrary[Long])
      a = a0.sorted
    } yield {
      val len = a.length
      for { key <- a ++ moreKeys }
        if (key > a(0) && key < a(len - 1))
          assert(RowPartitioner.findInterval(a, key) == naiveFindInterval(a, key))
      true
    }
    Prop.forAll(g).check()
  }
}
