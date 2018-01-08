package is.hail.distributedmatrix

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowPartitionerSuite extends TestNGSuite {

  @Test
  def testSmall() {
    val partitionStarts = Array[Long](0, 4, 5, 10)
    val partitionCounts = Array(4, 1, 5)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }
  
    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == 3)
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
  
  @Test
  def testSmallWithEmptyPartitions() {
    val partitionStarts = Array[Long](0, 0, 0, 4, 5, 5, 10, 10)
    val partitionCounts = Array(0, 0, 4, 1, 0, 5, 0)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }
  
    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == 7)
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
  
  @Test
  def testBig() {
    val partitionStarts = Array.tabulate(151)(i => 2 * i.toLong)  
    partitionStarts(10) -= 2
    partitionStarts(20) -= 1
    partitionStarts(30) += 1
    partitionStarts(40) += 2
    
    val nParts = partitionStarts.length - 1
    val partitionCounts = Array.tabulate(nParts)(pi => (partitionStarts(pi + 1) - partitionStarts(pi)).toInt)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }    

    val rp = RowPartitioner(partitionStarts)
    assert(rp.numPartitions == nParts)    
    assert((0 until 300).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
}
