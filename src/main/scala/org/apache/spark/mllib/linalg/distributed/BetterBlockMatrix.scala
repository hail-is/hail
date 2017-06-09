package org.apache.spark.mllib.linalg.distributed

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.internal.Logging

object BetterBlockMatrix {

  class BlockMatrixMultiplyRDD(l: BlockMatrix, r: BlockMatrix, deps: Seq[Dependency[_]], partitions: Array[Partition])
      extends RDD[((Int, Int), Matrix)](l.blocks.sparkContext, deps) {

    val lPartitioner = l.blocks.partitioner.get.asInstanceOf[GridPartitioner]
    val rPartitioner = r.blocks.partitioner.get.asInstanceOf[GridPartitioner]
    val nRows: Long = l.numRows()
    val nCols: Long = r.numCols()
    val rowsPerBlock: Int = l.rowsPerBlock
    val colsPerBlock: Int = r.colsPerBlock
    val rowBlocks = l.numRowBlocks
    val colBlocks = r.numColBlocks
    val nProducts = l.numColBlocks
    val rowsRemainder: Int = (nRows % rowsPerBlock).toInt
    val colsRemainder: Int = (nCols % colsPerBlock).toInt

    private def toDenseMatrix(x: Matrix): DenseMatrix = x match {
      case x: DenseMatrix => x
      case x: SparseMatrix => x.toDense
    }

    private def multiplyAccumulateWithExtension(x: DenseMatrix, y: DenseMatrix, result: DenseMatrix) {
      val x2 = if (result.numRows != x.numRows)
        toDenseMatrix(Matrices.vertcat(Array(x, Matrices.zeros(result.numRows - x.numRows, x.numCols))))
      else
        x

      val y2 = if (result.numCols != y.numCols)
        toDenseMatrix(Matrices.horzcat(Array(y, Matrices.zeros(y.numRows, result.numCols - y.numCols))))
      else
        y

      BLAS.gemm(1.0, x2, y2, 1.0, result)
    }

    private def block(bm: BlockMatrix, p: GridPartitioner, context: TaskContext, i: Int, j: Int): Matrix =
      bm.blocks.compute(IntPartition(p.getPartition((i, j))), context).toArray match {
        case Array((_, m)) => m
        case x => throw new RuntimeException(s"Expected array of length one, but got blocks: ${x.map(_._1).toSeq}, from $bm with partitioner $p.")
      }
    private def leftBlock(i: Int, j: Int, context: TaskContext): Matrix =
      block(l, lPartitioner, context, i, j)
    private def rightBlock(i: Int, j: Int, context: TaskContext): Matrix =
      block(r, rPartitioner, context, i, j)

    def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), Matrix)] = {
      val row = split.index % rowBlocks
      val col = split.index / rowBlocks
      val rowsInThisBlock: Int = (if (row + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
      val colsInThisBlock: Int = (if (col + 1 == colBlocks && colsRemainder != 0) colsRemainder else colsPerBlock)

      val finalResult = DenseMatrix.zeros(rowsInThisBlock, colsInThisBlock)
      var i = 0
      while (i < nProducts) {
        val leftMat = leftBlock(row, i, context)
        val rightMat = rightBlock(i, col, context)
        (leftMat, rightMat) match {
          case (x: DenseMatrix, y: DenseMatrix) =>
            multiplyAccumulateWithExtension(x, y, finalResult)
          case _ =>
             throw new SparkException(s"No support for multiplying: ${leftMat.getClass} by ${rightMat.getClass}.")
        }

        i += 1
      }

      Iterator.single(((row, col), finalResult))
    }

    protected def getPartitions: Array[Partition] =
      partitions

    /** Optionally overridden by subclasses to specify how they are partitioned. */
    @transient override val partitioner: Option[Partitioner] =
      Some(new GridPartitioner(rowBlocks, colBlocks, 1, 1))
  }

  case class IntPartition(index: Int) extends Partition { }

  private def gridPartition(m: BlockMatrix): (BlockMatrix, GridPartitioner) = {
    val p = GridPartitioner(m.numRowBlocks, m.numColBlocks)
    (new BlockMatrix(m.blocks.partitionBy(p), m.rowsPerBlock, m.colsPerBlock, m.numRows(), m.numCols()),
      p)
  }

  private def ensureGridPartitioning(m: BlockMatrix): (BlockMatrix, GridPartitioner) = l.blocks.partitioner match {
    case Some(gp: GridPartitioner) if (gp.rowsPerPart == 1 && gp.colsPerPart == 1) =>
      (l, gp)
    case Some(gp: GridPartitioner) =>
      logDebug(s"Repartitioning a matrix (slow), $l, with a grid partitioner that didn't have 1 block per partition, had: ${gp.rowsPerPart} x ${gp.colsPerPart}")
      gridPartition(l)
    case Some(p) =>
      logDebug(s"Repartitioning a matrix (slow), $l, with a non-grid partitioner: $p")
      gridPartition(l)
    case None =>
      logDebug(s"Partitioning a matrix (slow), $l, with a no partitioner")
      gridPartition(l)
  }

  def multiply(preL: BlockMatrix, preR: BlockMatrix): BlockMatrix = {
    require(preL.numCols() == preR.numRows(),
      s"""The number of columns of theleft matrix and the number of rows of the right
          matrix must be equal: ${l.numRows()} x ${l.numCols()}, ${r.numRows()}
          x ${r.numCols()}. If you think they should be equal, try setting the
          dimensions of A and B explicitly while initializing them.""".stripMargin)
    // FIXME: don't require same blocksize on each matrix
    require(preL.colsPerBlock == preR.rowsPerBlock,
      s"""The number of columns in blocks of the left matrix and the number of rows in
          blocks of the right matrix must be equal: ${l.rowsPerBlock} x
          ${l.colsPerBlock}, ${r.rowsPerBlock} x ${r.colsPerBlock}. Generally,
          all matrices should use square blocks of the same dimension.""".stripMargin)

    val (l, lPartitioner) = ensureGridPartitioning(preL)
    val (r, rPartitioner) = ensureGridPartitioning(preR)
    val rowBlocks = l.numRowBlocks
    val nProducts = l.numColBlocks
    val colBlocks = r.numColBlocks
    val deps = Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val row = partitionId % rowBlocks
          val deps = new Array[Int](nProducts)
          var i = 0
          while (i < nProducts) {
            deps(i) = lPartitioner.getPartition((row, i))
            i += 1
          }
          deps
        }
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val col = partitionId / rowBlocks
          val deps = new Array[Int](nProducts)
          var i = 0
          while (i < nProducts) {
            deps(i) = rPartitioner.getPartition((i, col))
            i += 1
          }
          deps
        }
      })
    val nParts = rowBlocks * colBlocks
    val parts = (0 until nParts).map(IntPartition.apply _).toArray[Partition]

    new BlockMatrix(new BlockMatrixMultiplyRDD(l, r, deps, parts), l.rowsPerBlock, r.colsPerBlock, l.numRows(), r.numCols())
  }

  class BlockMatrixTransposeRDD(m: BlockMatrix)
      extends RDD[((Int, Int), Matrix)](m.blocks.sparkContext, Seq[Dependency[_]](new OneToOneDependency(m.blocks))) {

    def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), Matrix)] = {
      m.blocks.compute(split, context).map { case ((i, j), m) => ((j, i), m.transpose) }
    }

    private val parts =
      (0 until m.numRowBlocks * m.numColBlocks).map(IntPartition.apply _).toArray[Partition]
    protected def getPartitions: Array[Partition] =
      // XXX: This is dumb why don't I have permission to access this?
      // m.blocks.getPartitions
      parts

    private val prevPartitioner = m.blocks.partitioner.get
    @transient override val partitioner: Option[Partitioner] =
      Some(new GridPartitioner(m.numColBlocks, m.numRowBlocks, 1, 1) {
        override def getPartition(key: Any) = key match {
          case i: Int => super.getPartition(i)
          case (i: Int, j: Int) =>
            prevPartitioner.getPartition((j, i))
          case (i: Int, j: Int, _: Int) =>
            prevPartitioner.getPartition((j, i))
          case _ =>
            throw new IllegalArgumentException(s"Unrecognized key: $key.")
        }
      })
  }

  def transpose(preM: BlockMatrix): BlockMatrix = {
    val (m, _) = ensureGridPartitioning(m)
    new BlockMatrix(new BlockMatrixTransposeRDD(m), m.colsPerBlock, m.rowsPerBlock, m.numCols(), m.numRows())
  }
}
