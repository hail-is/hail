package is.hail.methods

import is.hail.HailContext
import is.hail.expr.types._
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import is.hail.utils.fatal
import is.hail.variant.Locus

import scala.collection.mutable

class IndexedLocus(val index: Long, val contig: String, val pos: Int) {

  val locus = new Locus(contig, pos)

  def near(other: IndexedLocus, window: Int): Boolean = locus.near(other.locus, window)

}

object EntriesTableFilterByWindow {

  /* Input: block matrix with variants along both axes, and a table of indexed loci corresponding to those variants.
  Output: entries table that contains only entries where both variants are within the specified window distance.  */
  def apply(hc: HailContext, tbl: Table, bm: BlockMatrix, window: Int): Table = {

    val intervals = new VariantIntervals().computeIntervalsOfNearbyLociByIndex(tbl, window)
      .map { case (start: Long, end: Long) => Array(start, end) }.toArray

    val blocksToKeep = bm.gp.upperDiagonalBlocks(intervals)
    bm.entriesTable(hc, Some(blocksToKeep))
  }
}

class VariantIntervals() {

  private val contendersForCurrentInterval = new mutable.Queue[IndexedLocus]
  private var stackOfNearbyLocusIntervals = List[(Long, Long)]()

  // compute a set of indexed intervals of loci that are within the specified window distance of each other
  // assumes table is ordered by chromosome and position
  def computeIntervalsOfNearbyLociByIndex(tbl: Table, window: Int): List[(Long, Long)] = {
    checkSignature(tbl.signature)

    tbl.rdd.collect().foreach { r =>
      val curr = new IndexedLocus(r.get(0).asInstanceOf[Long], r.get(1).asInstanceOf[String], r.get(2).asInstanceOf[Int])

      if (contendersForCurrentInterval.isEmpty) {
        startNewInterval(curr)
      } else {
        contendersForCurrentInterval += curr
        val furthestBehind = getFurthestLocusWithinWindow(curr, window)
        if (contendersForCurrentInterval.lengthCompare(1) >= 0) {
          addInterval(curr.index, furthestBehind.index)
        }
      }
    }
    stackOfNearbyLocusIntervals
  }

  def checkSignature(signature: TStruct): Unit = {
    val expectedSignature = TStruct("index" -> TInt64(), "contig" -> TString(), "pos" -> TInt32())
    if (signature != expectedSignature) {
      fatal(s"Expected table to have signature $expectedSignature but found ${ signature }")
    }
  }

  def startNewInterval(curr: IndexedLocus): Unit = {
    stackOfNearbyLocusIntervals = (curr.index, curr.index) :: stackOfNearbyLocusIntervals
    contendersForCurrentInterval += curr
  }

  def getFurthestLocusWithinWindow(curr: IndexedLocus, window: Int): IndexedLocus = {
    var peek = contendersForCurrentInterval.front
    while (!curr.near(peek, window)) {
      contendersForCurrentInterval.dequeue()
      peek = contendersForCurrentInterval.front
    }
    peek
  }

  def addInterval(endOfNewInterval: Long, startOfNewInterval: Long): Unit = {

    val (startOfInterval, endOfInterval) = stackOfNearbyLocusIntervals.head

    // check for overlap with interval in stack
    if (startOfInterval == startOfNewInterval) {
      coalesceIntervals(startOfInterval, endOfInterval, startOfNewInterval, endOfNewInterval)
    } else {
      stackOfNearbyLocusIntervals = (startOfNewInterval, endOfNewInterval) :: stackOfNearbyLocusIntervals
    }
  }

  def coalesceIntervals(startOfInterval: Long, endOfInterval: Long,
    startOfNewInterval: Long, endOfNewInterval: Long): Unit = {

    stackOfNearbyLocusIntervals = stackOfNearbyLocusIntervals.tail
    stackOfNearbyLocusIntervals = (startOfInterval, endOfNewInterval) :: stackOfNearbyLocusIntervals
  }
}
