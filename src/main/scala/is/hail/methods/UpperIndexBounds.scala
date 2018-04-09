package is.hail.methods

import is.hail.HailContext
import is.hail.annotations.UnsafeIndexedSeq
import is.hail.expr.types._
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import is.hail.utils.fatal
import org.apache.spark.sql.Row

object UpperIndexBounds {

  def groupPositionsByContig(tbl: Table): Array[Array[Int]] = {
    val expectedSignature = TStruct("contig" -> TString(), "pos" -> TInt32())
    if (tbl.signature != expectedSignature) {
      fatal(s"Expected table to have signature $expectedSignature, but found ${ tbl.signature }.")
    }

    val convertUnsafeSeqToIntArray = (unsafe: UnsafeIndexedSeq) => unsafe.toArray.map(_.asInstanceOf[Row])
      .map(r => r.get(0).asInstanceOf[Int])

    val groupedPositions = tbl.keyBy("contig").groupByKey("positions").collect()
      .map(r => convertUnsafeSeqToIntArray(r.get(1).asInstanceOf[UnsafeIndexedSeq]))

    groupedPositions.foreach(scala.util.Sorting.quickSort)
    groupedPositions
  }

  // positions is non-decreasing, radius is non-negative
  // for each index i, compute the largest index j such that positions[j]-positions[i] <= radius
  def computeUpperIndexBounds(positions: Array[Int], radius: Int): Array[Int] = {
    var contenders = positions.zipWithIndex.tail

    val bounds = positions.zipWithIndex.map { case (pos, i) =>
      val largestJ = contenders.reverse.find { case (posj, j) => posj - pos <= radius }
      if (contenders.lengthCompare(1) > 0) {
        contenders = contenders.tail
      }
      largestJ.getOrElse((pos, i))._2
    }

    bounds
  }
  
  // compute index bounds for each array in groupedPositions, then shift the index bounds by the value of firstIndices
  def computeUpperIndexBounds(groupedPositions: Array[Array[Int]], radius: Int): Array[Long] = {
    val firstIndices = groupedPositions.map(_.length).scanLeft(0L)(_ + _)

    (firstIndices, groupedPositions).zipped.map {
      case (firstIndex, positions) => computeUpperIndexBounds(positions, radius).map(firstIndex + _)
    }.flatten
  }
  
  // compute which blocks of the block matrix have entries whose variants are within a certain radius of each other
  def computeBlocksWithinRadius(tbl: Table, bm: BlockMatrix, radius: Int): Array[Int] = {
    val groupedPositions = groupPositionsByContig(tbl)
    val upperIndexBounds = computeUpperIndexBounds(groupedPositions, radius)
    bm.gp.rectangularBlocks(upperIndexBounds.zipWithIndex.flatMap {
      case (j, i) => for {k <- i.toLong until j} yield Array(i, k, k + 1, k + 1)
    })
  }
  
  // get an entries table from a windowed block matrix, where window specifies the size of the window between variants
  def entriesTableFromWindowedBlockMatrix(hc: HailContext, tbl: Table, bm: BlockMatrix, window: Int): Table = {
    val blocksToKeep = computeBlocksWithinRadius(tbl, bm, window)
    bm.entriesTable(hc, Some(blocksToKeep))
  }
}
