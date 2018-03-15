package is.hail.rvd

import is.hail.annotations.UnsafeIndexedSeq
import is.hail.expr.types._
import is.hail.utils.Interval
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import org.testng.Assert._

class OrderedRVDPartitionerSuite extends TestNGSuite {
  @Test def testGetPartitionPK() {
    val partitioner =
      new OrderedRVDPartitioner(
        Array("A", "B"),
        TStruct(("A", TInt32()), ("C", TInt32()), ("B", TInt32())),
        UnsafeIndexedSeq(
          TArray(TInterval(TTuple(TInt32(), TInt32()), true), true),
          IndexedSeq(
            Interval(Row(1, 0), Row(4, 3), true, false),
            Interval(Row(4, 3), Row(7, 9), true, false),
            Interval(Row(7, 9), Row(10, 0), true, true)))
      )
    assert(partitioner.getPartitionPK(Row(0, 1, 3)) == 0)
    assert(partitioner.getPartitionPK(Row(2, 7, 5)) == 0)
    assert(partitioner.getPartitionPK(Row(4, 2, 1)) == 0)
    assert(partitioner.getPartitionPK(Row(4, 3, 5)) == 1)
    assert(partitioner.getPartitionPK(Row(7, 9, 7)) == 2)
    assert(partitioner.getPartitionPK(Row(11, 1, 42)) == 2)

    assert(partitioner.getPartitionPK(Row(4, 2)) == 0)
    assert(partitioner.getPartitionPK(Row(4, 3)) == 1)

    assert(partitioner.getPartitionPK(Row(2)) == 0)
    println(partitioner.getPartitionPK(Row(4)))
  }
}
