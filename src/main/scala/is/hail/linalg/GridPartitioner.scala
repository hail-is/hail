package is.hail.linalg

import org.apache.spark.Partitioner
import breeze.linalg.{DenseVector => BDV}
import is.hail.utils._


case class GridPartitioner(blockSize: Int, nRows: Long, nCols: Long, maybeSparse: Option[Array[Int]] = None) extends Partitioner {
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
  
  require(maybeSparse.forall(bis => bis.isEmpty ||
    (bis.isIncreasing && bis.head >= 0 && bis.last < maxNBlocks &&
      bis.length < maxNBlocks))) // a sparse block matrix cannot have all blocks present

  val lastBlockRowNRows: Int = indexBlockOffset(nRows - 1) + 1
  val lastBlockColNCols: Int = indexBlockOffset(nCols - 1) + 1
  
  def blockRowNRows(i: Int): Int = if (i < nBlockRows - 1) blockSize else lastBlockRowNRows
  def blockColNCols(j: Int): Int = if (j < nBlockCols - 1) blockSize else lastBlockColNCols

  def blockBlockRow(bi: Int): Int = bi % nBlockRows
  def blockBlockCol(bi: Int): Int = bi / nBlockRows

  def blockDims(bi: Int): (Int, Int) = (blockRowNRows(blockBlockRow(bi)), blockColNCols(blockBlockCol(bi)))
  
  def blockCoordinates(bi: Int): (Int, Int) = (blockBlockRow(bi), blockBlockCol(bi))

  def coordinatesBlock(i: Int, j: Int): Int = {
    require(0 <= i && i < nBlockRows, s"Block row $i out of range [0, $nBlockRows).")
    require(0 <= j && j < nBlockCols, s"Block column $j out of range [0, $nBlockCols).")
    i + j * nBlockRows
  }
  
  def filterBlocks(blocksToKeep: Array[Int]): (GridPartitioner, Array[Int]) = {
    require(blocksToKeep.isEmpty ||
      (blocksToKeep.isIncreasing && blocksToKeep.head >= 0 && blocksToKeep.last < maxNBlocks)) // could move into Some
    
    val (filteredBlocks, partsToKeep) = maybeSparse match {
      case Some(bis) =>
        val blocksToKeepSet = blocksToKeep.toSet
        bis.zipWithIndex.filter { case (bi, _) => blocksToKeepSet(bi) }.unzip
      case None => (blocksToKeep, blocksToKeep)
    }
    
    val filteredGP =
      if (partsToKeep.length == numPartitions)
        this
      else
        GridPartitioner(blockSize, nRows, nCols, Some(filteredBlocks))

    (filteredGP, partsToKeep)
  }
  
  def intersectBlocks(that: GridPartitioner): Option[Array[Int]] = {
    (maybeSparse, that.maybeSparse) match {
      case (Some(bis), Some(bis2)) => Some(bis.filter(bis2.toSet))
      case (Some(bis), None) => Some(bis)
      case (None, Some(bis2)) => Some(bis2)
      case (None, None) => None
    }
  }
  
  def unionBlocks(that: GridPartitioner): Option[Array[Int]] = {
    (maybeSparse, that.maybeSparse) match {
      case (Some(bis), Some(bis2)) =>
        val union = bis.union(bis2)
        scala.util.Sorting.quickSort(union)
        Some(union)
      case _ => None
    }
  }

  override val numPartitions: Int = maybeSparse match {
    case Some(bis) => bis.length
    case None =>
      assert(maxNBlocks < Int.MaxValue)
      maxNBlocks.toInt
  }
  
  val partBlock: Int => Int = maybeSparse match {
    case Some(bis) => pi =>
      assert(pi >= 0 && pi < bis.length)
      bis(pi)
    case None => pi =>
      assert(pi >= 0 && pi < numPartitions)
      pi
  }
  
  val blockPart: Int => Int = maybeSparse match {
    case Some(bis) => bis.zipWithIndex.toMap.withDefaultValue(-1)
    case None => bi => bi
  }
  
  def partCoordinates(pi: Int): (Int, Int) = blockCoordinates(partBlock(pi))

  def coordinatesPart(i: Int, j: Int): Int = blockPart(coordinatesBlock(i, j))

  override def getPartition(key: Any): Int = key match {
    case (i: Int, j: Int) => coordinatesPart(i, j)
  }
  
  def transpose: (GridPartitioner, Int => Int) = {
    val gpT = GridPartitioner(blockSize, nCols, nRows)
    def transposeBI(bi: Int): Int = coordinatesBlock(gpT.blockBlockCol(bi), gpT.blockBlockRow(bi))
    maybeSparse match {
      case Some(bis) =>
        val (biTranspose, piTranspose) = bis.map(transposeBI).zipWithIndex.sortBy(_._1).unzip
        val inverseTransposePI = piTranspose.zipWithIndex.sortBy(_._1).map(_._2)
        
        (GridPartitioner(blockSize, nCols, nRows, Some(biTranspose)), inverseTransposePI)
      case None => (gpT, transposeBI)
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
  
  // returns increasing array of all blocks intersecting the diagonal band consisting of
  //   all elements with lower <= jj - ii <= upper
  def bandedBlocks(lower: Long, upper: Long): Array[Int] = {
    require(lower <= upper)
    
    val lowerBlock = java.lang.Math.floorDiv(lower, blockSize).toInt
    val upperBlock = java.lang.Math.floorDiv(upper + blockSize - 1, blockSize).toInt

    (for { j <- 0 until nBlockCols
           i <- ((j - upperBlock) max 0) to
                ((j - lowerBlock) min (nBlockRows - 1))
    } yield (j * nBlockRows) + i).toArray
  }

  // returns increasing array of all blocks intersecting the rectangle [firstRow, lastRow] x [firstCol, lastCol]
  def rectangularBlocks(firstRow: Long, lastRow: Long, firstCol: Long, lastCol: Long): Array[Int] = {
    require(firstRow >= 0 && lastRow < nRows)
    require(firstCol >= 0 && lastCol < nCols)
    
    if (firstRow > lastRow || firstCol > lastCol)
      return Array.empty[Int]
    
    val firstBlockRow = indexBlockIndex(firstRow)
    val lastBlockRow = indexBlockIndex(lastRow)
    val firstBlockCol = indexBlockIndex(firstCol)
    val lastBlockCol = indexBlockIndex(lastCol)
    
    (for { j <- firstBlockCol to lastBlockCol
           i <- firstBlockRow to lastBlockRow
    } yield (j * nBlockRows) + i).toArray
  }

  // returns increasing array of all blocks intersecting the union of rectangles
  def rectangularBlocks(rectangles: Array[Array[Long]]): Array[Int] = {
    require(rectangles.forall(r => r.length == 4))
    val rects = rectangles.foldLeft(Set[Int]())((s, r) => s ++ rectangularBlocks(r(0), r(1), r(2), r(3))).toArray    
    scala.util.Sorting.quickSort(rects)
    rects
  }
}
