package is.hail.distributedmatrix

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowPartitionerSuite extends TestNGSuite {

  @Test
  def testSmall() {
    val partitionStarts = Array[Long](0, 4, 5, 10)
    val rangeBounds = Array[Long](3, 4)
    assert(RowPartitioner.rangeBoundsFromStarts(partitionStarts).sameElements(rangeBounds))

    val partitionCounts = Array(4, 1, 5)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }
    val rp = new RowPartitioner(rangeBounds)
    assert(rp.numPartitions == 3)    
    assert((0 until 10).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
  
  @Test
  def testBig() {
    val partitionStarts = Array.tabulate(151)(i => 2 * i.toLong)  
    partitionStarts(10) -= 2
    partitionStarts(20) -= 1
    partitionStarts(30) += 1
    partitionStarts(40) += 2
    
    val rangeBounds = Array.tabulate(149)(i => 2 * i.toLong + 1)
    rangeBounds(9) -= 2
    rangeBounds(19) -= 1
    rangeBounds(29) += 1
    rangeBounds(39) += 2
    
    assert(RowPartitioner.rangeBoundsFromStarts(partitionStarts).sameElements(rangeBounds))
    
    val nParts = partitionStarts.length - 1
    val partitionCounts = Array.tabulate(nParts)(pi => (partitionStarts(pi + 1) - partitionStarts(pi)).toInt)
    val keyPart = partitionCounts.zipWithIndex.flatMap{ case (count, pi) => Array.fill(count)(pi) }    

    val rp = RowPartitioner.fromPartitionStarts(partitionStarts)
    assert(rp.numPartitions == nParts)
    assert((0 until 300).forall(i => keyPart(i) == rp.getPartition(i.toLong)))
  }
}
