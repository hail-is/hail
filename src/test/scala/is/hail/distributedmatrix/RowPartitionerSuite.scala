package is.hail.distributedmatrix

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowPartitionerSuite extends TestNGSuite {

  @Test
  def test() {
    val partitionStarts = Array[Long](0, 4, 5, 10)
    val partitionCounts = Array(4, 1, 5)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }
  
    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == 3)
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
  
  @Test
  def testWithEmptyPartitions() {
    val partitionStarts = Array[Long](0, 0, 0, 4, 5, 5, 8, 10, 10)
    val partitionCounts = Array(0, 0, 4, 1, 0, 3, 2, 0)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }
    
    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == 8)   
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
}
