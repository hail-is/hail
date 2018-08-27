package is.hail.rvd

import is.hail.annotations.UnsafeIndexedSeq
import is.hail.expr.types._
import is.hail.utils.Interval
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class OrderedRVDPartitionerSuite extends TestNGSuite {
  val kType = TStruct(("A", TInt32()), ("B", TInt32()), ("C", TInt32()))
  val partitioner =
    new OrderedRVDPartitioner(kType,
      Array(
        Interval(Row(1, 0), Row(4, 3), true, false),
        Interval(Row(4, 3), Row(7, 9), true, false),
        Interval(Row(7, 11), Row(10, 0), true, true))
    )

  @Test def testExtendKey() {
    val p = new OrderedRVDPartitioner(TStruct(("A", TInt32()), ("B", TInt32())),
      Array(
        Interval(Row(1, 0), Row(4, 3), true, true),
        Interval(Row(4, 3), Row(4, 3), true, true),
        Interval(Row(4, 3), Row(7, 9), true, false),
        Interval(Row(7, 11), Row(10, 0), true, true))
      )
    val extended = p.extendKey(kType)
    assert(extended.rangeBounds sameElements Array(
      Interval(Row(1, 0), Row(4, 3), true, true),
      Interval(Row(4, 3), Row(7, 9), false, false),
      Interval(Row(7, 11), Row(10, 0), true, true))
    )
  }

  @Test def testGetPartitionWithPartitionKeys() {
    assert(partitioner.getSafePartitionLowerBound(Row(-1, 7)) == 0)
    assert(partitioner.getSafePartitionUpperBound(Row(-1, 7)) == 0)

    assert(partitioner.getSafePartitionLowerBound(Row(4, 2)) == 0)
    assert(partitioner.getSafePartitionUpperBound(Row(4, 2)) == 1)

    assert(partitioner.getSafePartitionLowerBound(Row(4, 3)) == 1)
    assert(partitioner.getSafePartitionUpperBound(Row(4, 3)) == 2)

    assert(partitioner.getSafePartitionLowerBound(Row(5, -10259)) == 1)
    assert(partitioner.getSafePartitionUpperBound(Row(5, -10259)) == 2)

    assert(partitioner.getSafePartitionLowerBound(Row(7, 9)) == 2)
    assert(partitioner.getSafePartitionUpperBound(Row(7, 9)) == 2)

    assert(partitioner.getSafePartitionLowerBound(Row(12, 19)) == 3)
    assert(partitioner.getSafePartitionUpperBound(Row(12, 19)) == 3)
  }

  @Test def testGetPartitionWithLargerKeys() {
    assert(partitioner.getSafePartitionLowerBound(Row(0, 1, 3)) == 0)
    assert(partitioner.getSafePartitionUpperBound(Row(0, 1, 3)) == 0)

    assert(partitioner.getSafePartitionLowerBound(Row(2, 7, 5)) == 0)
    assert(partitioner.getSafePartitionUpperBound(Row(2, 7, 5)) == 1)

    assert(partitioner.getSafePartitionLowerBound(Row(4, 2, 1, 2.7, "bar")) == 0)

    assert(partitioner.getSafePartitionLowerBound(Row(7, 9, 7)) == 2)
    assert(partitioner.getSafePartitionUpperBound(Row(7, 9, 7)) == 2)

    assert(partitioner.getSafePartitionLowerBound(Row(11, 1, 42)) == 3)
  }

   @Test def testGetPartitionPKWithSmallerKeys() {
     assert(partitioner.getSafePartitionLowerBound(Row(2)) == 0)
     assert(partitioner.getSafePartitionUpperBound(Row(2)) == 1)

     assert(partitioner.getSafePartitionLowerBound(Row(4)) == 0)
     assert(partitioner.getSafePartitionUpperBound(Row(4)) == 2)

     assert(partitioner.getSafePartitionLowerBound(Row(11)) == 3)
     assert(partitioner.getSafePartitionUpperBound(Row(11)) == 3)
   }

  @Test def testGetPartitionRange() {
    assert(partitioner.getPartitionRange(Interval(Row(3, 4), Row(7, 11), true, true)) == Seq(0, 1, 2))
    assert(partitioner.getPartitionRange(Interval(Row(3, 4), Row(7, 9), true, false)) == Seq(0, 1))
    assert(partitioner.getPartitionRange(Interval(Row(-1, 7), Row(0, 9), true, false)) == Seq())
  }

  @Test def testGetSafePartitionKeyRange() {
    assert(partitioner.getSafePartitionKeyRange(Row(0, 0)).isEmpty)
    assert(partitioner.getSafePartitionKeyRange(Row(7, 10)).isEmpty)
    assert(partitioner.getSafePartitionKeyRange(Row(7, 11)) == Range.inclusive(2, 2))
  }

  @Test def testGenerateDisjoint() {
    val intervals = Array(
        Interval(Row(1, 0, 4), Row(4, 3, 2), true, false),
        Interval(Row(4, 3, 5), Row(7, 9, 1), true, false),
        Interval(Row(7, 11, 3), Row(10, 0, 1), true, true),
        Interval(Row(11, 0, 2), Row(11, 0, 15), false, true),
        Interval(Row(11, 0, 15), Row(11, 0, 20), true, false))

    val p3 = OrderedRVDPartitioner.generate(Array("A", "B", "C"), kType, intervals)
    assert(p3.satisfiesAllowedOverlap(2))
    assert(p3.rangeBounds sameElements
      Array(
        Interval(Row(1, 0, 4), Row(4, 3, 2), true, false),
        Interval(Row(4, 3, 5), Row(7, 9, 1), true, false),
        Interval(Row(7, 11, 3), Row(10, 0, 1), true, true),
        Interval(Row(11, 0, 2), Row(11, 0, 15), false, true),
        Interval(Row(11, 0, 15), Row(11, 0, 20), false, false))
    )

    val p2 = OrderedRVDPartitioner.generate(Array("A", "B"), kType, intervals)
    assert(p2.satisfiesAllowedOverlap(1))
    assert(p2.rangeBounds sameElements
      Array(
        Interval(Row(1, 0, 4), Row(4, 3), true, true),
        Interval(Row(4, 3), Row(7, 9, 1), false, false),
        Interval(Row(7, 11, 3), Row(10, 0, 1), true, true),
        Interval(Row(11, 0, 2), Row(11, 0, 20), false, false))
    )

    val p1 = OrderedRVDPartitioner.generate(Array("A"), kType, intervals)
    assert(p1.satisfiesAllowedOverlap(0))
    assert(p1.rangeBounds sameElements
      Array(
        Interval(Row(1, 0, 4), Row(4), true, true),
        Interval(Row(4), Row(7), false, true),
        Interval(Row(7), Row(10, 0, 1), false, true),
        Interval(Row(11, 0, 2), Row(11, 0, 20), false, false))
    )
  }

  @Test def testIntersect() {
    val kType = TStruct(("key", TInt32()))
    val left =
      new OrderedRVDPartitioner(kType,
        Array(
          Interval(Row(1), Row(10), true, false),
          Interval(Row(12), Row(13), true, false),
          Interval(Row(14), Row(19), true, false))
      )
    val right =
      new OrderedRVDPartitioner(kType,
        Array(
          Interval(Row(1), Row(4), true, false),
          Interval(Row(4), Row(5), true, false),
          Interval(Row(7), Row(16), true, true),
          Interval(Row(19), Row(20), true, true))
      )
    assert(left.intersect(right).rangeBounds sameElements
      Array(
        Interval(Row(1), Row(4), true, false),
        Interval(Row(4), Row(5), true, false),
        Interval(Row(7), Row(10), true, false),
        Interval(Row(12), Row(13), true, false),
        Interval(Row(14), Row(16), true, true)
      )
    )
  }
}
