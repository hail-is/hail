package is.hail.linalg

import is.hail.TestUtils._
import is.hail.collection.compat.immutable.ArraySeq

import org.junit.jupiter.api.Test
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Prop.forAll

class RowPartitionerSuite {
  @Test
  def testGetPartition(): Unit = {
    val partitionStarts = ArraySeq[Long](0, 0, 0, 4, 5, 5, 8, 10, 10)
    val partitionCounts = Array(0, 0, 4, 1, 0, 3, 2, 0)
    val keyPart = partitionCounts.zipWithIndex.flatMap { case (count, pi) => Array.fill(count)(pi) }

    val rp = RowPartitioner(partitionStarts)
    assertEq(rp.numPartitions, 8)
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }

  @Test def testFindInterval(): Unit = {
    def naiveFindInterval(a: IndexedSeq[Long], key: Long): Int = {
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

    check(forAll(arbitrary[ArraySeq[Long]] map { _.sorted }) { a =>
      a.isEmpty || {
        (a ++ moreKeys).foreach { key =>
          assert(
            !(key > a.head && key < a.last) ||
              RowPartitioner.findInterval(a, key) == naiveFindInterval(a, key)
          )
        }
        true
      }
    })
  }
}
