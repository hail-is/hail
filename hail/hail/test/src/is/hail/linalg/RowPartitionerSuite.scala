package is.hail.linalg

import is.hail.collection.compat.immutable.ArraySeq

import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Prop._

class RowPartitionerSuite extends munit.ScalaCheckSuite {
  test("GetPartition") {
    val partitionStarts = ArraySeq[Long](0, 0, 0, 4, 5, 5, 8, 10, 10)
    val partitionCounts = Array(0, 0, 4, 1, 0, 3, 2, 0)
    val keyPart = partitionCounts.zipWithIndex.flatMap { case (count, pi) => Array.fill(count)(pi) }

    val rp = RowPartitioner(partitionStarts)
    assertEquals(rp.numPartitions, 8)
    (0 until 10).foreach(i => assertEquals(rp.getPartition(i.toLong), keyPart(i)))
  }

  property("FindInterval") {
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

    forAll(arbitrary[ArraySeq[Long]].map(_.sorted)) { a =>
      (a.nonEmpty) ==> {
        (a ++ moreKeys).forall { key =>
          !(key > a.head && key < a.last) ||
          RowPartitioner.findInterval(a, key) == naiveFindInterval(a, key)
        }
      }
    }
  }
}
