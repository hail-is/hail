package is.hail.methods

import is.hail.HailContext
import is.hail.expr.types._
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import is.hail.utils.fatal
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object UpperIndexBounds {

  def groupPositionsByContig(tbl: Table): RDD[Array[Int]] = {

    if (tbl.keySignature.size < 1) {
      fatal(s"Expected table to have at least 1 key, but found ${ tbl.keySignature.size } keys.")
    }
    if (tbl.valueSignature.size != 1 || tbl.valueSignature.types(0) != TInt32()) {
      fatal(s"Expected table to have 1 field of type int32, but found ${ tbl.valueSignature.size } field(s) of " +
        s"type(s) ${ tbl.valueSignature.types.mkString(", ") }.")
    }

    tbl.groupByKey("positions").rdd.map(row => row.get(1).asInstanceOf[Vector[Row]].toArray
      .map(row => row.get(0).asInstanceOf[Int]))
  }

  // positions is non-decreasing, radius is non-negative
  // for each index i, compute the largest index j such that positions[j]-positions[i] <= radius
  def computeUpperIndexBounds(positions: Array[Int], radius: Int): Array[Int] = {

    val bounds = for {i <- positions.indices} yield {
      val maxPossiblePositionInsideRadius = positions(i) + radius
      var largestJ = i
      var j = i
      while (j < positions.length) {
        if (positions(j) <= maxPossiblePositionInsideRadius) {
          largestJ = j
          j += 1
        } else {
          j = positions.length
        }
      }
      largestJ
    }

    bounds.toArray
  }

  def shiftUpperIndexBounds(upperIndexBounds: RDD[Array[Int]]): Array[Long] = {
    val firstIndices = upperIndexBounds.map(_.length).collect().scanLeft(0L)(_ + _)

    (firstIndices, upperIndexBounds.collect()).zipped.map {
      case (firstIndex, bounds) => bounds.map(firstIndex + _)
    }.flatten
  }

  // compute which blocks of the block matrix have entries whose variants are within a certain radius of each other
  // assumes a symmetric matrix, so only blocks above the diagonal are included
  def computeBlocksWithinRadiusAndAboveDiagonal(tbl: Table, bm: BlockMatrix, radius: Int): Array[Int] = {

    val groupedPositions = groupPositionsByContig(tbl)

    val upperIndexBounds = groupedPositions.map(positions => {
      scala.util.Sorting.quickSort(positions)
      computeUpperIndexBounds(positions, radius)
    })

    val shiftedUpperIndexBounds = shiftUpperIndexBounds(upperIndexBounds)

    bm.gp.rectangularBlocks(shiftedUpperIndexBounds.zipWithIndex.flatMap {
      case (j, i) => if (i == j) None else Some(Array(i, i, i + 1, j))
    })
  }

  // get an entries table from a windowed block matrix, where window specifies the size of the window between variants
  def entriesTableFromWindowedBlockMatrix(hc: HailContext, tbl: Table, bm: BlockMatrix, window: Int): Table = {
    val blocksToKeep = computeBlocksWithinRadiusAndAboveDiagonal(tbl, bm, window)
    bm.entriesTable(hc, Some(blocksToKeep))
  }
}
