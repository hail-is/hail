package org.apache.spark.mllib.linalg.distributed

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.internal.Logging

object BetterBlockMatrix extends Logging {

  class BlockMatrixMultiplyRDD(l: BlockMatrix, r: BlockMatrix)
      extends RDD[((Int, Int), Matrix)](l.blocks.sparkContext, Nil) {

    private val lPartitioner = l.blocks.partitioner.get.asInstanceOf[GridPartitioner]
    private val lPartitions = l.blocks.partitions
    private val rPartitioner = r.blocks.partitioner.get.asInstanceOf[GridPartitioner]
    private val rPartitions = r.blocks.partitions
    private val nRows: Long = l.numRows()
    private val nCols: Long = r.numCols()
    private val rowsPerBlock: Int = l.rowsPerBlock
    private val colsPerBlock: Int = r.colsPerBlock
    private val rowBlocks = l.numRowBlocks
    private val colBlocks = r.numColBlocks
    private val nProducts = l.numColBlocks
    private val rowsRemainder: Int = (nRows % rowsPerBlock).toInt
    private val colsRemainder: Int = (nCols % colsPerBlock).toInt
    private val nParts = rowBlocks * colBlocks

    override def getDependencies: Seq[Dependency[_]] =
      Array[Dependency[_]](
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

    private def toDenseMatrix(x: Matrix): DenseMatrix = x match {
      case x: DenseMatrix => x
      case x: SparseMatrix => x.toDense
    }

    private def block(bm: BlockMatrix, bmPartitions: Array[Partition], p: GridPartitioner, context: TaskContext, i: Int, j: Int): Matrix = {
      val it = bm.blocks.compute(bmPartitions(p.getPartition((i, j))), context)
      val r = it.next()
      assert(!it.hasNext, s"Expected iterator of size one, but got blocks: ${r +: it.toSeq}, from $bm with partitioner $p. ${bm.blocks.toDebugString}")
      r._2
    }

    private def leftBlock(i: Int, j: Int, context: TaskContext): Matrix =
      block(l, lPartitions, lPartitioner, context, i, j)
    private def rightBlock(i: Int, j: Int, context: TaskContext): Matrix =
      block(r, rPartitions, rPartitioner, context, i, j)

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
            BLAS.gemm(1.0, x, y, 1.0, result)
          case _ =>
            throw new SparkException(s"No support for multiplying: ${leftMat.getClass} by ${rightMat.getClass}.")
        }

        i += 1
      }

      Iterator.single(((row, col), finalResult))
    }

    val parts = (0 until nParts).map(IntPartition.apply _).toArray[Partition]
    protected def getPartitions: Array[Partition] =
      parts

    /** Optionally overridden by subclasses to specify how they are partitioned. */
    @transient override val partitioner: Option[Partitioner] =
      Some(new GridPartitioner(rowBlocks, colBlocks, 1, 1))
  }

  case class IntPartition(index: Int) extends Partition { }

  private def gridPartition(m: BlockMatrix): BlockMatrix = {
    val p = new GridPartitioner(m.numRowBlocks, m.numColBlocks, 1, 1)
    new BlockMatrix(m.blocks.partitionBy(p), m.rowsPerBlock, m.colsPerBlock, m.numRows(), m.numCols())
  }

  private def ensureGridPartitioning(m: BlockMatrix): BlockMatrix = m.blocks.partitioner match {
    case Some(gp: GridPartitioner) if (gp.rowsPerPart == 1 && gp.colsPerPart == 1) =>
      m
    case Some(gp: GridPartitioner) =>
      logDebug(s"Repartitioning a matrix (slow), $m, with a grid partitioner that didn't have 1 block per partition, had: ${gp.rowsPerPart} x ${gp.colsPerPart}")
      gridPartition(m)
    case Some(p) =>
      logDebug(s"Repartitioning a matrix (slow), $m, with a non-grid partitioner: $p")
      gridPartition(m)
    case None =>
      logDebug(s"Partitioning a matrix (slow), $m, with a no partitioner")
      gridPartition(m)
  }

  def multiply(preL: BlockMatrix, preR: BlockMatrix): BlockMatrix = {
    require(preL.numCols() == preR.numRows(),
      s"""The number of columns of theleft matrix and the number of rows of the right
          matrix must be equal: ${preL.numRows()} x ${preL.numCols()}, ${preR.numRows()}
          x ${preR.numCols()}. If you think they should be equal, try setting the
          dimensions of A and B explicitly while initializing them.""".stripMargin)
    // FIXME: don't require same blocksize on each matrix
    require(preL.colsPerBlock == preR.rowsPerBlock,
      s"""The number of columns in blocks of the left matrix and the number of rows in
          blocks of the right matrix must be equal: ${preL.rowsPerBlock} x
          ${preL.colsPerBlock}, ${preR.rowsPerBlock} x ${preR.colsPerBlock}. Generally,
          all matrices should use square blocks of the same dimension.""".stripMargin)

    val l = ensureGridPartitioning(preL)
    val r = ensureGridPartitioning(preR)
    new BlockMatrix(new BlockMatrixMultiplyRDD(l, r), l.rowsPerBlock, r.colsPerBlock, l.numRows(), r.numCols())
  }

  class BlockMatrixTransposeRDD(m: BlockMatrix)
      extends RDD[((Int, Int), Matrix)](m.blocks.sparkContext, Seq[Dependency[_]](new OneToOneDependency(m.blocks))) {

    def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), Matrix)] = {
      val it = m.blocks.compute(split, context).map { case ((i, j), m) => ((j, i), m.transpose) }
      val r = it.next()
      assert(!it.hasNext, s"Expected iterator of size one, but got blocks: ${r +: it.toSeq}, from $m with partitioner ${m.blocks.partitioner}. ${m.blocks.toDebugString}")
      Iterator.single(r)
    }

    private val parts =
      (0 until m.numRowBlocks * m.numColBlocks).map(IntPartition.apply _).toArray[Partition]
    protected def getPartitions: Array[Partition] =
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
    val m = ensureGridPartitioning(preM)
    new BlockMatrix(new BlockMatrixTransposeRDD(m), m.colsPerBlock, m.rowsPerBlock, m.numCols(), m.numRows())
  }
}
