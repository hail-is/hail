package org.apache.spark.mllib.linalg.distributed

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

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

    private def fmaDenseMatricesWithExtension(x: DenseMatrix, y: DenseMatrix, result: DenseMatrix) {
      val x2 = if (finalResult.numRows != x.numRows)
        toDenseMatrix(Matrices.vertcat(Array(x, Matrices.zeros(finalResult.numRows - x.numRows, x.numCols))))
      else
        x

      val y2 = if (finalResult.numCols != y.numCols)
        toDenseMatrix(Matrices.horzcat(Array(y, Matrices.zeros(y.numRows, finalResult.numCols - y.numCols))))
      else
        y

      BLAS.gemm(1.0, x2, y2, 1.0, finalResult)
    }

    private def block(bm: BlockMatrix, p: GridPartitioner, i: Int, j: Int): Matrix =
      bm.blocks.compute(IntPartition(p.getPartition((i, j))), context).toArray match {
        case Array((_, m)) => m
        case x => throw new RuntimeException(s"Expected array of length one, but got blocks: ${x.map(_._1).toSeq}, from $bm with partitioner $p.")
      }
    private def leftBlock(i: Int, j: Int): Matrix =
      block(l, lPartitioner, i, j)
    private def rightBlock(i: Int, j: Int): Matrix =
      block(r, rPartitioner, i, j)

    def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), Matrix)] = {
      val row = split.index % rowBlocks
      val col = split.index / rowBlocks
      val rowsInThisBlock: Int = (if (row + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
      val colsInThisBlock: Int = (if (col + 1 == colBlocks && colsRemainder != 0) colsRemainder else colsPerBlock)

      val finalResult = DenseMatrix.zeros(rowsInThisBlock, colsInThisBlock)
      var i = 0
      while (i < nProducts) {
        (leftBlock(row, i), rightBlock(i, col)) match {
          case (x: DenseMatrix, y: DenseMatrix) =>
            fmaDenseMatricesWithExtension(x, y, finalResult)
          case (x: DenseMatrix, y: SparseMatrix) =>
            fmaDenseMatricesWithExtension(x, toDenseMatrix(y), finalResult)
          case (x: SparseMatrix, y: DenseMatrix) =>
            fmaDenseMatricesWithExtension(toDenseMatrix(x), y, finalResult)
          case _ =>
             throw new SparkException(s"I only support Dense * Dense, recieved: ${leftMat.getClass}, ${rightMat.getClass}.")
        }

        i += 1
      }

      Array(((row, col), finalResult)).iterator
    }

    /**
      * Implemented by subclasses to return the set of partitions in this RDD. This method will only
      * be called once, so it is safe to implement a time-consuming computation in it.
      *
      * The partitions in this array must satisfy the following property:
      *   `rdd.partitions.zipWithIndex.forall { case (partition, index) => partition.index == index }`
      */
    protected def getPartitions: Array[Partition] =
      partitions

    /** Optionally overridden by subclasses to specify how they are partitioned. */
    @transient override val partitioner: Option[Partitioner] =
      Some(new GridPartitioner(rowBlocks, colBlocks, 1, 1))
  }

  case class IntPartition(index: Int) extends Partition

  def multiply(l: BlockMatrix, r: BlockMatrix): BlockMatrix = {
    require(l.numCols() == r.numRows(), "The number of columns of A and the number of rows " +
      s"of B must be equal. A.numCols: ${l.numCols()}, B.numRows: ${r.numRows()}. If you " +
      "think they should be equal, try setting the dimensions of A and B explicitly while " +
      "initializing them.")

    // FIXME: don't require same blocksize on each matrix
    require(l.colsPerBlock == r.rowsPerBlock)
    require(l.numColBlocks == r.numRowBlocks)

    if (l.colsPerBlock == r.rowsPerBlock) {
      val lPartitioner = l.blocks.partitioner.get.asInstanceOf[GridPartitioner]
      val rPartitioner = r.blocks.partitioner.get.asInstanceOf[GridPartitioner]
      val rowBlocks = l.numRowBlocks
      val colBlocks = r.numColBlocks
      val nProducts = l.numColBlocks
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
    } else {
      throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
        s"A.colsPerBlock: ${l.colsPerBlock}, B.rowsPerBlock: ${r.rowsPerBlock}")
    }
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

  def transpose(m: BlockMatrix): BlockMatrix = {
    require(!m.blocks.partitioner.isEmpty)
    new BlockMatrix(new BlockMatrixTransposeRDD(m), m.colsPerBlock, m.rowsPerBlock, m.numCols(), m.numRows())
  }
}
