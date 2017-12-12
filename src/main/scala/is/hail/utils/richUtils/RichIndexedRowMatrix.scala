package is.hail.utils.richUtils

import org.apache.spark._
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import is.hail.utils._
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix

object RichIndexedRowMatrix {
  private def seqOp(gp: GridPartitioner)
    (block: Array[Double], row: (Int, Int, Int, Array[Double])): Array[Double] = {

    val (i, j, ii, rowSegment) = row
    val nRowsInBlock = gp.blockRowNRows(i)
    val block2 = if (block == null) new Array[Double](nRowsInBlock * gp.blockColNCols(j)) else block

    var jj = 0
    while (jj < rowSegment.length) {
      block2(jj * nRowsInBlock + ii) = rowSegment(jj)
      jj += 1
    }
    block2
  }

  private def combOp(l: Array[Double], r: Array[Double]): Array[Double] = {
    if (l == null)
      r
    else if (r == null)
      l
    else {
      var k = 0
      while (k < l.length) {
        l(k) += r(k)
        k += 1
      }
      l
    }
  }
}

class RichIndexedRowMatrix(irm: IndexedRowMatrix) {
  import RichIndexedRowMatrix._

  def toHailBlockMatrix(blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {
    require(blockSize > 0, s"blockSize must be greater than 0. blockSize: $blockSize")

    val nRows = irm.numRows()
    val nCols = irm.numCols()
    val gp = GridPartitioner(blockSize, nRows, nCols)
    val nBlockCols = gp.nBlockCols

    val blocks = irm.rows.flatMap { ir =>
      val i = (ir.index / blockSize).toInt
      val ii = (ir.index % blockSize).toInt
      val entireRow = ir.vector.toArray
      val rowSegments = new Array[((Int, Int), (Int, Int, Int, Array[Double]))](nBlockCols)

      var j = 0
      while (j < nBlockCols) {
        val nColsInBlock = gp.blockColNCols(j)
        val rowSegmentInBlock = new Array[Double](nColsInBlock)
        System.arraycopy(entireRow, j * blockSize, rowSegmentInBlock, 0, nColsInBlock)
        rowSegments(j) = ((i, j), (i, j, ii, rowSegmentInBlock))
        j += 1
      }

      rowSegments.iterator
    }.aggregateByKey(null: Array[Double], gp)(seqOp(gp), combOp)
      .mapValuesWithKey { case ((i, j), data) =>
        new BDM[Double](gp.blockRowNRows(i), gp.blockColNCols(j), data)
    }

    new BlockMatrix(new EmptyPartitionIsAZeroMatrixRDD(blocks), blockSize, nRows, nCols)
  }
}

private class EmptyPartitionIsAZeroMatrixRDD(blocks: RDD[((Int, Int), BDM[Double])])
    extends RDD[((Int, Int), BDM[Double])](blocks.sparkContext, Seq[Dependency[_]](new OneToOneDependency(blocks))) {
  @transient val gp: GridPartitioner = (blocks.partitioner: @unchecked) match {
    case Some(p: GridPartitioner) => p
  }

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val p = split.asInstanceOf[BlockPartition]
    val it = blocks.iterator(split, context)
    if (it.hasNext)
      it
    else
      Iterator.single(p.coords -> BDM.zeros[Double](p.blockSize, p.blockSize))
  }

  protected def getPartitions: Array[Partition] =
    blocks.partitions.map { p =>
      new BlockPartition(p.index, gp.blockSize, gp.blockCoordinates(p.index))
    }.toArray

  @transient override val partitioner: Option[Partitioner] =
    Some(gp)
}

private class BlockPartition(val index: Int, val blockSize: Int, val coords: (Int, Int)) extends Partition {}
