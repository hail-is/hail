package is.hail.utils.richUtils

import org.apache.spark._
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.distributedmatrix._
import is.hail.utils._
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.json4s.jackson

object RichIndexedRowMatrix {
  private def seqOp(truncatedBlockRow: Int, truncatedBlockCol: Int, excessRows: Int, excessCols: Int, blockSize: Int)
    (block: Array[Double], row: (Int, Int, Int, Array[Double])): Array[Double] = row match {
    case (i, j, ii, a) =>
      val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
      val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize
      val block2 = if (block == null) new Array[Double](rowsInBlock * colsInBlock) else block

      var jj = 0
      while (jj < a.length) {
        block2(jj * rowsInBlock + ii) = a(jj)
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

class RichIndexedRowMatrix(indexedRowMatrix: IndexedRowMatrix) {
  import RichIndexedRowMatrix._

  def toHailBlockMatrix(blockSize: Int = BlockMatrix.defaultBlockSize): BlockMatrix = {
    require(blockSize > 0,
      s"blockSize needs to be greater than 0. blockSize: $blockSize")

    val rows = indexedRowMatrix.numRows()
    val cols = indexedRowMatrix.numCols()
    val partitioner = GridPartitioner(blockSize, rows, cols)
    val colPartitions = partitioner.colPartitions
    // NB: if excessRows == 0, we never reach the truncatedBlockRow
    val truncatedBlockRow = (rows / blockSize).toInt
    val truncatedBlockCol = (cols / blockSize).toInt
    val excessRows = (rows % blockSize).toInt
    val excessCols = (cols % blockSize).toInt
    val fullColPartitions = if (excessCols == 0) colPartitions else colPartitions - 1

    val blocks = indexedRowMatrix.rows.flatMap { ir =>
      val i = (ir.index / blockSize).toInt
      val ii = (ir.index % blockSize).toInt
      val a = ir.vector.toArray
      val grouped = new Array[((Int, Int), (Int, Int, Int, Array[Double]))](colPartitions)

      var j = 0
      while (j < fullColPartitions) {
        val group = new Array[Double](blockSize)
        grouped(j) = ((i, j), (i, j, ii, group))
        System.arraycopy(a, j * blockSize, group, 0, blockSize)
        j += 1
      }
      if (excessCols > 0) {
        val group = new Array[Double](excessCols)
        grouped(j) = ((i, j), (i, j, ii, group))
        System.arraycopy(a, j * blockSize, group, 0, excessCols)
      }

      grouped.iterator
    }.aggregateByKey(null: Array[Double], partitioner)(
      seqOp(truncatedBlockRow, truncatedBlockCol, excessRows, excessCols, blockSize), combOp)
      .mapValuesWithKey { case ((i, j), a) =>
        val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
        val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize
        new BDM[Double](rowsInBlock, colsInBlock, a)
    }

    new BlockMatrix(new EmptyPartitionIsAZeroMatrixRDD(blocks), blockSize, rows, cols)
  }
  
  def writeAsBlockMatrix(uri: String, blockSize: Int) {
    val rows = indexedRowMatrix.numRows()
    val cols = indexedRowMatrix.numCols()
    
    val hadoop = indexedRowMatrix.rows.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)
    
    // write blocks
    hadoop.mkDir(uri + "/parts")
    val gp = GridPartitioner(blockSize, rows, cols)
    val blockCount = new WriteBlocksRDD(indexedRowMatrix, uri, gp).reduce(_ + _)
    assert(blockCount == gp.numPartitions)
    info(s"Wrote all $blockCount blocks of $rows x $cols matrix with block size $blockSize.")
    
    // write metadata
    hadoop.writeDataFile(uri + BlockMatrix.metadataRelativePath) { os =>
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, rows, cols, transposed = false),
        os)
    }
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
