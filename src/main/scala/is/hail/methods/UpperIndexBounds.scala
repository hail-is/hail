package is.hail.methods

import is.hail.expr.types._
import is.hail.linalg.{BlockMatrix, GridPartitioner}
import is.hail.table.Table
import is.hail.utils.{fatal, plural}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object UpperIndexBounds {

  def groupPositionsByKey(tbl: Table): RDD[Array[Int]] = {

    if (tbl.valueSignature.size != 1 || tbl.valueSignature.types(0) != TInt32()) {
      fatal(s"Expected table to have one value field of type int32, but found ${ tbl.valueSignature.size } value " +
        s"${ plural(tbl.valueSignature.size, "field") } of ${ plural(tbl.valueSignature.size, "type") } " +
        s"${ tbl.valueSignature.types.mkString(", ") }.")
    }

    val fieldIndex = tbl.valueFieldIdx(0)

    tbl.groupByKey("positions").rdd.map(row => row.get(fieldIndex).asInstanceOf[Vector[Row]].toArray
      .map(row => row.get(0).asInstanceOf[Int]))
  }

  // positions is non-decreasing, radius is non-negative
  // for each index i, compute the largest index j such that positions[j]-positions[i] <= radius
  def computeUpperIndexBounds(positions: Array[Int], radius: Int): Array[Int] = {

    val n = positions.length
    val bounds = new Array[Int](n)
    var j = 0

    for (i <- positions.indices) {
      val maxPosition = positions(i) + radius
      while (j < n && positions(j) <= maxPosition) {
        j += 1
      }
      j -= 1
      bounds(i) = j
    }

    bounds
  }

  def shiftUpperIndexBounds(upperIndexBounds: RDD[Array[Int]]): Array[Long] = {
    val firstIndices = upperIndexBounds.map(_.length).collect().scanLeft(0L)(_ + _)

    (firstIndices, upperIndexBounds.collect()).zipped.flatMap {
      case (firstIndex, bounds) => bounds.map(firstIndex + _)
    }
  }

  /* computes the minimum set of blocks necessary to cover all pairs of indices (i, j) such that i <= j and 
  position[j] - position[i] <= radius and i <= j.  If includeDiagonal=false, require i < j rather than i <= j. */
  def computeBlocksWithinRadiusAndAboveDiagonal(tbl: Table, gp: GridPartitioner, radius: Int, includeDiagonal: Boolean): Array[Int] = {

    val groupedPositions = groupPositionsByKey(tbl)

    val relativeUpperIndexBounds = groupedPositions.map(positions => {
      scala.util.Sorting.quickSort(positions)
      computeUpperIndexBounds(positions, radius)
    })

    val absoluteUpperIndexBounds = shiftUpperIndexBounds(relativeUpperIndexBounds)

    if (includeDiagonal) {
      gp.rectangularBlocks(absoluteUpperIndexBounds.zipWithIndex.map {
        case (j, i) => Array(i, i, i, j)
      })
    } else {
      gp.rectangularBlocks(absoluteUpperIndexBounds.zipWithIndex.flatMap {
        case (j, i) => if (i == j) None else Some(Array(i, i, i + 1, j))
      })
    }
  }
}
