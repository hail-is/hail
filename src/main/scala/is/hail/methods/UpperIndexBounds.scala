package is.hail.methods

import is.hail.expr.types._
import is.hail.linalg.GridPartitioner
import is.hail.table.Table
import is.hail.utils.plural
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object UpperIndexBounds {
  def groupPositionsByKey(t: Table): RDD[Array[Int]] = {
    require(t.valueSignature.size == 1 && t.valueSignature.types(0) == TInt32(),
      s"Expected table to have one value field of type int32, but found ${ t.valueSignature.size } value " +
        s"${ plural(t.valueSignature.size, "field") } of ${ plural(t.valueSignature.size, "type") } " +
        s"${ t.valueSignature.types.mkString(", ") }.")

    val groupedTable = t.groupByKey("positions")
    val positionsIndex = groupedTable.valueFieldIdx(0)

    groupedTable.rdd.map(
      _.get(positionsIndex).asInstanceOf[Vector[Row]].toArray.map(_.get(0).asInstanceOf[Int]))
  }

  // positions is non-decreasing, radius is non-negative
  // for each index i, compute the largest j such that positions[k]-positions[i] <= radius
  // for all k in [i, j)
  // FIXME: do in Python or add to expression language, then do block filtering with sparsify_row_intervals
  def computeUpperIndexBounds(positions: Array[Int], radius: Int): Array[Int] = {
    val n = positions.length
    val bounds = new Array[Int](n)
    var j = 0

    for (i <- positions.indices) {
      val maxPosition = positions(i) + radius
      while (j < n && positions(j) <= maxPosition) {
        j += 1
      }
      bounds(i) = j
      j -= 1
    }

    bounds
  }

  def shiftUpperIndexBounds(upperIndexBounds: RDD[Array[Int]]): Array[Long] = {
    val firstIndices = upperIndexBounds.map(_.length).collect().scanLeft(0L)(_ + _)

    (firstIndices, upperIndexBounds.collect()).zipped.flatMap {
      case (firstIndex, bounds) => bounds.map(firstIndex + _)
    }
  }

  /* computes the minimum set of blocks necessary to cover all pairs of indices (i, j) such that key[i] == key[j], 
  i <= j, and position[j] - position[i] <= radius.  If includeDiagonal=false, require i < j rather than i <= j. */
  def computeCoverByUpperTriangularBlocks(
    t: Table, gp: GridPartitioner, radius: Int, includeDiagonal: Boolean): Array[Int] = {

    val relativeUpperIndexBounds = groupPositionsByKey(t).map(positions => {
      scala.util.Sorting.quickSort(positions)
      computeUpperIndexBounds(positions, radius)
    })

    val absoluteUpperIndexBounds = shiftUpperIndexBounds(relativeUpperIndexBounds)
    assert(absoluteUpperIndexBounds.length == gp.nRows)

    if (includeDiagonal)
      gp.rowIntervalsBlocks((0L until gp.nRows).toArray, absoluteUpperIndexBounds)
    else
      gp.rowIntervalsBlocks((1L until (gp.nRows + 1)).toArray, absoluteUpperIndexBounds)
  }
}
