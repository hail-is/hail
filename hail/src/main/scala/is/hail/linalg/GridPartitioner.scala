package is.hail.linalg

import is.hail.utils._

import scala.collection.mutable

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.Partitioner

/** BLOCKS ARE NUMBERED COLUMN MAJOR
  *
  * @param blockSize
  * @param nRows
  * @param nCols
  * @param partitionIndexToBlockIndex
  *   If exists, matrix is sparse and this contains a list of indices of blocks that are not all
  *   zero
  */
case class GridPartitioner(
  blockSize: Int,
  nRows: Long,
  nCols: Long,
  partitionIndexToBlockIndex: Option[IndexedSeq[Int]] = None,
) extends Partitioner {
  if (nRows == 0)
    fatal("block matrix must have at least one row")

  if (nCols == 0)
    fatal("block matrix must have at least one column")

  require(nRows <= Int.MaxValue.toLong * blockSize)
  require(nCols <= Int.MaxValue.toLong * blockSize)

  def indexBlockIndex(index: Long): Int = (index / blockSize).toInt

  def indexBlockOffset(index: Long): Int = (index % blockSize).toInt

  val nBlockRows: Int = indexBlockIndex(nRows - 1) + 1
  val nBlockCols: Int = indexBlockIndex(nCols - 1) + 1

  val maxNBlocks: Long = nBlockRows.toLong * nBlockCols

  if (
    !partitionIndexToBlockIndex.forall(bis =>
      bis.isEmpty ||
        (bis.isIncreasing && bis.head >= 0 && bis.last < maxNBlocks &&
          bis.length < maxNBlocks)
    )
  ) // a block-sparse matrix cannot have all blocks present
    throw new IllegalArgumentException(
      s"requirement failed: Sparse blocks sequence was ${partitionIndexToBlockIndex.toIndexedSeq}, max was $maxNBlocks"
    )

  val blockToPartitionMap =
    partitionIndexToBlockIndex.map(_.zipWithIndex.toMap.withDefaultValue(-1))

  val lastBlockRowNRows: Int = indexBlockOffset(nRows - 1) + 1
  val lastBlockColNCols: Int = indexBlockOffset(nCols - 1) + 1

  def blockRowNRows(i: Int): Int = if (i < nBlockRows - 1) blockSize else lastBlockRowNRows
  def blockColNCols(j: Int): Int = if (j < nBlockCols - 1) blockSize else lastBlockColNCols

  def blockBlockRow(bi: Int): Int = bi % nBlockRows
  def blockBlockCol(bi: Int): Int = bi / nBlockRows

  def blockDims(bi: Int): (Int, Int) =
    (blockRowNRows(blockBlockRow(bi)), blockColNCols(blockBlockCol(bi)))

  def nBlocks: Int = partitionIndexToBlockIndex.map(_.length).getOrElse(nBlockRows * nBlockCols)

  def blockCoordinates(bi: Int): (Int, Int) = (blockBlockRow(bi), blockBlockCol(bi))

  def coordinatesBlock(i: Int, j: Int): Int = {
    require(0 <= i && i < nBlockRows, s"Block row $i out of range [0, $nBlockRows).")
    require(0 <= j && j < nBlockCols, s"Block column $j out of range [0, $nBlockCols).")
    i + j * nBlockRows
  }

  def intersect(that: GridPartitioner): GridPartitioner = {
    copy(partitionIndexToBlockIndex =
      (partitionIndexToBlockIndex, that.partitionIndexToBlockIndex) match {
        case (Some(bis), Some(bis2)) => Some(bis.filter(bis2.toSet))
        case (Some(bis), None) => Some(bis)
        case (None, Some(bis2)) => Some(bis2)
        case (None, None) => None
      }
    )
  }

  def union(that: GridPartitioner): GridPartitioner = {
    copy(partitionIndexToBlockIndex =
      (partitionIndexToBlockIndex, that.partitionIndexToBlockIndex) match {
        case (Some(bis), Some(bis2)) =>
          val union = (bis ++ bis2).distinct
          if (union.length == maxNBlocks)
            None
          else
            Some(union.sorted)
        case _ => None
      }
    )
  }

  override val numPartitions: Int = partitionIndexToBlockIndex match {
    case Some(bis) => bis.length
    case None =>
      assert(maxNBlocks < Int.MaxValue)
      maxNBlocks.toInt
  }

  def partitionToBlock(pi: Int): Int = partitionIndexToBlockIndex match {
    case Some(bis) =>
      assert(pi >= 0 && pi < bis.length)
      bis(pi)
    case None =>
      assert(pi >= 0 && pi < numPartitions)
      pi
  }

  def blockToPartition(blockId: Int): Int = blockToPartitionMap match {
    case Some(bpMap) => bpMap(blockId)
    case None => blockId
  }

  def partCoordinates(pi: Int): (Int, Int) = blockCoordinates(partitionToBlock(pi))

  def coordinatesPart(i: Int, j: Int): Int = blockToPartition(coordinatesBlock(i, j))

  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => coordinatesPart(i, j)
  }

  /** @return
    *   A transposed GridPartitioner and a function that maps partitions in the new transposed
    *   partitioner to the parent partitions in the old partitioner.
    */
  def transpose: (GridPartitioner, Int => Int) = {
    val gpT = GridPartitioner(blockSize, nCols, nRows)
    partitionIndexToBlockIndex match {
      case Some(bis) =>
        def transposeBI(bi: Int): Int =
          gpT.coordinatesBlock(this.blockBlockCol(bi), this.blockBlockRow(bi))

        val (partIdxTToBlockIdxT, partIdxTToPartIdx) =
          bis.map(transposeBI).zipWithIndex.sortBy(_._1).unzip
        val transposedPartitionIndicesToParentPartitions = partIdxTToPartIdx.apply(_)

        (
          GridPartitioner(blockSize, nCols, nRows, Some(partIdxTToBlockIdxT)),
          transposedPartitionIndicesToParentPartitions,
        )
      case None =>
        def transposedBlockIndicesToParentBlocks(bi: Int) =
          this.coordinatesBlock(gpT.blockBlockCol(bi), gpT.blockBlockRow(bi))
        (gpT, transposedBlockIndicesToParentBlocks)
    }
  }

  def vectorOnBlockRow(v: BDV[Double], i: Int): BDV[Double] = {
    val firstRow = i * blockSize
    v(firstRow until firstRow + blockRowNRows(i))
  }

  def vectorOnBlockCol(v: BDV[Double], j: Int): BDV[Double] = {
    val firstCol = j * blockSize
    v(firstCol until firstCol + blockColNCols(j))
  }

  def maybeBlockRows(): Option[IndexedSeq[Int]] =
    partitionIndexToBlockIndex match {
      case Some(bis) =>
        val bisRow = bis.map(blockBlockRow).distinct
        if (bisRow.length < nBlockRows) Some(bisRow) else None
      case None => None
    }

  def maybeBlockCols(): Option[IndexedSeq[Int]] =
    partitionIndexToBlockIndex match {
      case Some(bis) =>
        val bisCol = bis.map(blockBlockCol).distinct
        if (bisCol.length < nBlockCols) Some(bisCol) else None
      case None => None
    }

  // returns increasing array of all blocks intersecting the diagonal band consisting of
  //   all elements with lower <= jj - ii <= upper
  def bandBlocks(lower: Long, upper: Long): Array[Int] = {
    require(lower <= upper)

    val lowerBlock = java.lang.Math.floorDiv(lower, blockSize).toInt
    val upperBlock = java.lang.Math.floorDiv(upper + blockSize - 1, blockSize).toInt

    (for {
      j <- 0 until nBlockCols
      i <- ((j - upperBlock) max 0) to
        ((j - lowerBlock) min (nBlockRows - 1))
    } yield (j * nBlockRows) + i).toArray
  }

  // returns increasing array of all blocks intersecting the rectangle
  // [r(0), r(1)) x [r(2), r(3)), i.e. [startRow, stopRow) x [startCol, stopCol)
  // rectangle checked in Python
  def rectangleBlocks(r: Array[Long]): Array[Int] = {
    val startBlockRow = indexBlockIndex(r(0))
    val stopBlockRow = java.lang.Math.floorDiv(r(1) - 1, blockSize).toInt + 1
    val startBlockCol = indexBlockIndex(r(2))
    val stopBlockCol = java.lang.Math.floorDiv(r(3) - 1, blockSize).toInt + 1

    (for {
      j <- startBlockCol until stopBlockCol
      i <- startBlockRow until stopBlockRow
    } yield (j * nBlockRows) + i).toArray
  }

  // returns increasing array of all blocks intersecting the union of rectangles
  // rectangles checked in Python
  def rectanglesBlocks(rectangles: Array[Array[Long]]): Array[Int] = {
    val blocks = rectangles.foldLeft(mutable.Set[Int]())((s, r) => s ++= rectangleBlocks(r)).toArray
    scala.util.Sorting.quickSort(blocks)
    blocks
  }

  // starts, stops checked in Python
  def rowIntervalsBlocks(starts: Array[Long], stops: Array[Long]): Array[Int] = {
    val rectangles = starts.grouped(blockSize).zip(stops.grouped(blockSize))
      .zipWithIndex
      .flatMap { case ((startsInBlockRow, stopsInBlockRow), blockRow) =>
        val nRowsInBlockRow = blockRowNRows(blockRow)
        var minStart = Long.MaxValue
        var maxStop = Long.MinValue
        var ii = 0
        while (ii < nRowsInBlockRow) {
          val start = startsInBlockRow(ii)
          val stop = stopsInBlockRow(ii)
          if (start < stop) {
            minStart = minStart min start
            maxStop = maxStop max stop
          }
          ii += 1
        }
        if (minStart < maxStop) {
          val row = blockRow * blockSize
          Some(Array(row, row + 1, minStart, maxStop))
        } else
          None
      }.toArray

    rectanglesBlocks(rectangles)
  }
}
