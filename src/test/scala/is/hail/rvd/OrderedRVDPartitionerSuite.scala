package is.hail.rvd

import is.hail.annotations.UnsafeIndexedSeq
import is.hail.expr.types._
import is.hail.utils.Interval
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class OrderedRVDPartitionerSuite extends TestNGSuite {
  val partitioner =
    new OrderedRVDPartitioner(
      Array("A", "B"),
      TStruct(("A", TInt32()), ("C", TInt32()), ("B", TInt32())),
        Array(
          Interval(Row(1, 0), Row(4, 3), true, false),
          Interval(Row(4, 3), Row(7, 9), true, false),
          Interval(Row(7, 9), Row(10, 0), true, true))
    )

  @Test def testGetPartitionPKWithPartitionKeys() {
    assert(partitioner.getPartitionPK(Row(-1, 7)) == 0)
    assert(partitioner.getPartitionPK(Row(4, 2)) == 0)
    assert(partitioner.getPartitionPK(Row(4, 3)) == 1)
    assert(partitioner.getPartitionPK(Row(5, -10259)) == 1)
    assert(partitioner.getPartitionPK(Row(7, 8)) == 1)
    assert(partitioner.getPartitionPK(Row(7, 9)) == 2)
    assert(partitioner.getPartitionPK(Row(10, 0)) == 2)
    assert(partitioner.getPartitionPK(Row(12, 19)) == 2)
  }

  @Test def testGetPartitionPKWithLargerKeys() {
    assert(partitioner.getPartitionPK(Row(0, 1, 3)) == 0)
    assert(partitioner.getPartitionPK(Row(2, 7, "foo")) == 0)
    assert(partitioner.getPartitionPK(Row(4, 2, 1, 2.7, "bar")) == 0)
    assert(partitioner.getPartitionPK(Row(4, 3, 5)) == 1)
    assert(partitioner.getPartitionPK(Row(7, 9, 7)) == 2)
    assert(partitioner.getPartitionPK(Row(11, 1, 42)) == 2)
  }

  // @Test def testGetPartitionPKWithSmallerKeys() {
  //   assert(partitioner.getPartitionPK(Row(2)) == 0)
  // }
}
