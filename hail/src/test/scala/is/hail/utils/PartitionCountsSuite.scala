package is.hail.utils

import is.hail.utils.PartitionCounts._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class PartitionCountsSuite extends TestNGSuite {

  @Test def testHeadPCs() = {
    for (
      ((a, n), b) <- Seq(
        (IndexedSeq(0L), 5L) -> IndexedSeq(0L),
        (IndexedSeq(4L, 5L, 6L), 1L) -> IndexedSeq(1L),
        (IndexedSeq(4L, 5L, 6L), 6L) -> IndexedSeq(4L, 2L),
        (IndexedSeq(4L, 5L, 6L), 9L) -> IndexedSeq(4L, 5L),
        (IndexedSeq(4L, 5L, 6L), 10L) -> IndexedSeq(4L, 5L, 1L),
        (IndexedSeq(4L, 5L, 6L), 15L) -> IndexedSeq(4L, 5L, 6L),
        (IndexedSeq(4L, 5L, 6L), 20L) -> IndexedSeq(4L, 5L, 6L),
      )
    )
      assert(getHeadPCs(a, n) == b, s"getHeadPartitionCounts($a, $n)")
  }

  @Test def testTailPCs() = {
    for (
      ((a, n), b) <- Seq(
        (IndexedSeq(0L), 5L) -> IndexedSeq(0L),
        (IndexedSeq(4L, 5L, 6L), 1L) -> IndexedSeq(1L),
        (IndexedSeq(4L, 5L, 6L), 6L) -> IndexedSeq(6L),
        (IndexedSeq(4L, 5L, 6L), 9L) -> IndexedSeq(3L, 6L),
        (IndexedSeq(4L, 5L, 6L), 10L) -> IndexedSeq(4L, 6L),
        (IndexedSeq(4L, 5L, 6L), 15L) -> IndexedSeq(4L, 5L, 6L),
        (IndexedSeq(4L, 5L, 6L), 20L) -> IndexedSeq(4L, 5L, 6L),
      )
    ) {
      assert(getTailPCs(a, n) == b, s"getTailPartitionCounts($a, $n)")
      assert(
        getTailPCs(a, n) == getHeadPCs(a.reverse, n).reverse,
        s"getTailPartitionCounts($a, $n) via head",
      )
    }
  }

  @Test def testIncrementalPCSubset() = {
    val pcs = Array(0L, 0L, 5L, 6L, 4L, 3L, 3L, 3L, 2L, 1L)

    def headOffset(n: Long) =
      incrementalPCSubsetOffset(n, 0 until pcs.length)(_.map(pcs))

    for (n <- 0L until pcs.sum) {
      val PCSubsetOffset(i, nKeep, nDrop) = headOffset(n)
      val total = (0 to i).map(j => if (j == i) nKeep else pcs(j)).sum
      assert(nKeep + nDrop == pcs(i))
      assert(total == n)
    }

    def tailOffset(n: Long) =
      incrementalPCSubsetOffset(n, (0 until pcs.length).reverse)(_.map(pcs))

    for (n <- 0L until pcs.sum) {
      val PCSubsetOffset(i, nKeep, nDrop) = tailOffset(n)
      val total = (i to (pcs.length - 1)).map(j => if (j == i) nKeep else pcs(j)).sum
      assert(nKeep + nDrop == pcs(i))
      assert(total == n)
    }
  }
}
