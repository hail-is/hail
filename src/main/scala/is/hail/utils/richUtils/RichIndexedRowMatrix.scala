package is.hail.utils.richUtils

import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, GridPartitioner, IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD

/**
  * Created by johnc on 3/23/17.
  */


class RichIndexedRowMatrix(rows: RDD[IndexedRow],
  private var nRows: Long,
  private var nCols: Int) extends IndexedRowMatrix(rows, nRows, nCols) {

  def toBlockMatrixPlus(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    /*
    PLAN:
    1. For each row, partition into ceil(numCols/colsPerBlock) of size colsPerBlock, aware that last partition may be smaller.
    2.
    */
    require(rowsPerBlock > 0,
      s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
    require(colsPerBlock > 0,
      s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")
    val m = numRows().toInt
    val n = numCols().toInt

    val temp = rows.flatMap({ir =>
      val blockRow = ir.index / rowsPerBlock
      val rowInBlock = ir.index % rowsPerBlock

      val partitionedArray: Iterator[Array[Double]] = ir.vector.toArray.grouped(colsPerBlock)
      val partitionedArrayWithIndices = partitionedArray.zipWithIndex

      partitionedArrayWithIndices.map(tuple =>
        tuple match { case (values, blockColumn) =>
          ((blockRow.toInt, blockColumn), (rowInBlock.toInt, values))})
    })

    val rdd: RDD[((Int, Int), Matrix)] = temp.groupByKey(new GridPartitioner(m, n, rowsPerBlock, colsPerBlock)).map(tuple =>
      tuple match {case (gridcoords, itr: Iterable[(Int, Array[Double])]) =>
        val array = new Array[Double](rowsPerBlock * colsPerBlock)
        itr.foreach(element => element match { case (rowWithinBlock, values) =>
          values.copyToArray(array, rowWithinBlock * n)
        })
        (gridcoords, new DenseMatrix(rowsPerBlock, colsPerBlock, array))
      })
    new BlockMatrix(rdd, rowsPerBlock, colsPerBlock)
  }
}

/**
  * A grid partitioner, which uses a regular grid to partition coordinates.
  *
  * @param rows Number of rows.
  * @param cols Number of columns.
  * @param rowsPerPart Number of rows per partition, which may be less at the bottom edge.
  * @param colsPerPart Number of columns per partition, which may be less at the right edge.
  */
private class GridPartitioner(
  val rows: Int,
  val cols: Int,
  val rowsPerPart: Int,
  val colsPerPart: Int) extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  private val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  private val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  /**
    * Returns the index of the partition the input coordinate belongs to.
    *
    * @param key The partition id i (calculated through this method for coordinate (i, j) in
    *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
    *            the inner index used in multiplication. k is ignored in computing partitions.
    * @return The index of the partition, which the coordinate belongs to.
    */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => i
      case (i: Int, j: Int) =>
        getPartitionId(i, j)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rows: java.lang.Integer,
      cols: java.lang.Integer,
      rowsPerPart: java.lang.Integer,
      colsPerPart: java.lang.Integer)
  }
}

private object GridPartitioner {

  /** Creates a new [[GridPartitioner]] instance. */
  def apply(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int): GridPartitioner = {
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }

  /** Creates a new [[GridPartitioner]] instance with the input suggested number of partitions. */
  def apply(rows: Int, cols: Int, suggestedNumPartitions: Int): GridPartitioner = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions)
    val rowsPerPart = math.round(math.max(scale * rows, 1.0)).toInt
    val colsPerPart = math.round(math.max(scale * cols, 1.0)).toInt
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }
}
