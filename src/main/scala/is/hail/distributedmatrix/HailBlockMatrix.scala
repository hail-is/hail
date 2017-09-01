package is.hail.distributedmatrix

import breeze.linalg.{DenseMatrix => BDM}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import is.hail.utils._
import org.apache.spark._

class HailBlockMatrix(val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val rows: Long,
  val cols: Long) extends Serializable {

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[HailGridPartitioner])

  val partitioner = blocks.partitioner.get.asInstanceOf[HailGridPartitioner]

  def transpose(): HailBlockMatrix =
    new HailBlockMatrix(new HailBlockMatrixTransposeRDD(this), blockSize, cols, rows)

  def add(other: HailBlockMatrix): HailBlockMatrix =
    blockMap2(other, _ + _)

  def subtract(other: HailBlockMatrix): HailBlockMatrix =
    blockMap2(other, _ - _)

  def pointwiseMultiply(other: HailBlockMatrix): HailBlockMatrix =
    blockMap2(other, _ :* _)

  def pointwiseDivide(other: HailBlockMatrix): HailBlockMatrix =
    blockMap2(other, _ :/ _)

  def multiply(other: HailBlockMatrix): HailBlockMatrix =
    new HailBlockMatrix(new HailBlockMatrixMultiplyRDD(this, other), blockSize, rows, other.cols)

  def cache(): this.type = {
    blocks.cache()
    this
  }

  def persist(storageLevel: StorageLevel): this.type = {
    blocks.persist(storageLevel)
    this
  }

  def toLocalMatrix(): BDM[Double] = {
    require(this.rows < Int.MaxValue, "The number of rows of this matrix should be less than " +
      s"Int.MaxValue. Currently numRows: ${this.rows}")
    require(this.cols < Int.MaxValue, "The number of columns of this matrix should be less than " +
      s"Int.MaxValue. Currently numCols: ${this.cols}")
    require(this.rows * this.cols < Int.MaxValue, "The length of the values array must be " +
      s"less than Int.MaxValue. Currently rows * cols: ${this.rows * this.cols}")
    val rows = this.rows.toInt
    val cols = this.cols.toInt
    val localBlocks = blocks.collect()
    val values = new Array[Double](rows * cols)
    var bi = 0
    while (bi < localBlocks.length) {
      val ((blocki, blockj), m) = localBlocks(bi)
      val ioffset = blocki * blockSize
      val joffset = blockj * blockSize
      m.foreachPair { case ((i, j), v) =>
        values((joffset + j) * rows + ioffset + i) = v
      }
      bi += 1
    }
    new BDM(rows, cols, values)
  }

  def blockMap(op: BDM[Double] => BDM[Double]): HailBlockMatrix =
    new HailBlockMatrix(blocks.mapValues(op), blockSize, rows, cols)

  private def requireZippable(other: HailBlockMatrix) {
    require(rows == other.rows,
      s"must have same number of rows, but actually: ${rows}x${cols}, ${other.rows}x${other.cols}")
    require(cols == other.cols,
      s"must have same number of cols, but actually: ${rows}x${cols}, ${other.rows}x${other.cols}")
    require(blockSize == other.blockSize,
      s"blocks must be same size, but actually were ${blockSize}x${blockSize} and ${other.blockSize}x${other.blockSize}")
  }

  def blockMap2(other: HailBlockMatrix, op: (BDM[Double], BDM[Double]) => BDM[Double]): HailBlockMatrix = {
    requireZippable(other)
    new HailBlockMatrix(blocks.join(other.blocks).mapValues(op.tupled), blockSize, rows, cols)
  }

  def map(op: Double => Double): HailBlockMatrix = {
    val blocks2 = blocks.mapValues { m =>
      val src = m.data
      val dst = new Array[Double](src.length)
      var i = 0
      while (i < src.length) {
        dst(i) = op(src(i))
        i += 1
      }
      new BDM(m.rows, m.cols, dst)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map2(other: HailBlockMatrix, op: (Double, Double) => Double): HailBlockMatrix = {
    requireZippable(other)
    val blocks2 = blocks.join(other.blocks).mapValues { case (m1, m2) =>
      val src1 = m1.data
      val src2 = m2.data
      val dst = new Array[Double](src1.length)
      var i = 0
      while (i < src1.length) {
        dst(i) = op(src1(i), src2(i))
        i += 1
      }
      new BDM(m1.rows, m1.cols, dst)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map3(hbm2: HailBlockMatrix, hbm3: HailBlockMatrix, op: (Double, Double, Double) => Double): HailBlockMatrix = {
    requireZippable(hbm2)
    requireZippable(hbm3)
    val blocks2 = blocks.join(hbm2.blocks).join(hbm3.blocks).mapValues { case ((m1, m2), m3) =>
      val src1 = m1.data
      val src2 = m2.data
      val src3 = m3.data
      val dst = new Array[Double](src1.length)
      var i = 0
      while (i < src1.length) {
        dst(i) = op(src1(i), src2(i), src3(i))
        i += 1
      }
      new BDM(m1.rows, m1.cols, dst)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map4(hbm2: HailBlockMatrix, hbm3: HailBlockMatrix, hbm4: HailBlockMatrix, op: (Double, Double, Double, Double) => Double): HailBlockMatrix = {
    requireZippable(hbm2)
    requireZippable(hbm3)
    requireZippable(hbm4)
    val blocks2 = blocks.join(hbm2.blocks).join(hbm3.blocks).join(hbm4.blocks).mapValues { case (((m1, m2), m3), m4) =>
      val src1 = m1.data
      val src2 = m2.data
      val src3 = m3.data
      val src4 = m4.data
      val dst = new Array[Double](src1.length)
      var i = 0
      while (i < src1.length) {
        dst(i) = op(src1(i), src2(i), src3(i), src4(i))
        i += 1
      }
      new BDM(m1.rows, m1.cols, dst)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def mapWithIndex(op: (Long, Long, Double) => Double): HailBlockMatrix = {
    val blockSize = this.blockSize
    val blocks2 = blocks.mapValuesWithKey { case ((blocki, blockj), m) =>
      val iprefix = blocki.toLong * blockSize
      val jprefix = blockj.toLong * blockSize
      val size = m.cols * m.rows
      val result = new Array[Double](size)
      var j = 0
      while (j < m.cols) {
        var i = 0
        while (i < m.rows) {
          result(i + j*m.rows) = op(iprefix + i, jprefix + j, m(i, j))
          i += 1
        }
        j += 1
      }
      new BDM(m.rows, m.cols, result)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map2WithIndex(other: HailBlockMatrix, op: (Long, Long, Double, Double) => Double): HailBlockMatrix = {
    requireZippable(other)
    val blockSize = this.blockSize
    val blocks2 = blocks.join(other.blocks).mapValuesWithKey { case ((blocki, blockj), (m1, m2)) =>
      val iprefix = blocki.toLong * blockSize
      val jprefix = blockj.toLong * blockSize
      val size = m1.cols * m1.rows
      val result = new Array[Double](size)
      var j = 0
      while (j < m1.cols) {
        var i = 0
        while (i < m1.rows) {
          result(i + j*m1.rows) = op(iprefix + i, jprefix + j, m1(i, j), m2(i, j))
          i += 1
        }
        j += 1
      }
      new BDM(m1.rows, m1.cols, result)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

}

private class HailBlockMatrixTransposeRDD(m: HailBlockMatrix)
    extends RDD[((Int, Int), BDM[Double])](m.blocks.sparkContext, Seq[Dependency[_]](new OneToOneDependency(m.blocks))) {
  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
    m.blocks.iterator(split, context).map { case ((i, j), m) => ((j, i), m.t) }

  protected def getPartitions: Array[Partition] =
    m.blocks.partitions

  private val prevPartitioner = m.partitioner
  @transient override val partitioner: Option[Partitioner] =
    Some(HailGridPartitioner(m.rows, m.cols, m.blockSize, transposed=true))
}

private class HailBlockMatrixMultiplyRDD(l: HailBlockMatrix, r: HailBlockMatrix)
    extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {
  require(l.cols == r.rows,
    s"inner dimension must match, but given: ${l.rows}x${l.cols}, ${r.rows}x${r.cols}")
  require(l.blockSize == r.blockSize,
    s"blocks must be same size, but actually were ${l.blockSize}x${l.blockSize} and ${r.blockSize}x${r.blockSize}")

  private val lPartitioner = l.partitioner
  private val lPartitions = l.blocks.partitions
  private val rPartitioner = r.partitioner
  private val rPartitions = r.blocks.partitions
  private val rows = l.rows
  private val cols = r.cols
  private val blockSize = l.blockSize
  private val rowBlocks = lPartitioner.rowPartitions
  private val colBlocks = rPartitioner.colPartitions
  private val nProducts = lPartitioner.colPartitions
  private val rowsRemainder = (rows % blockSize).toInt
  private val colsRemainder = (cols % blockSize).toInt
  private val nParts = rowBlocks * colBlocks

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val row = _partitioner.blockRowIndex(partitionId)
          val deps = new Array[Int](nProducts)
          var i = 0
          while (i < nProducts) {
            deps(i) = lPartitioner.partitionIdFromBlockIndices(row, i)
            i += 1
          }
          deps
        }
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val col = _partitioner.blockColIndex(partitionId)
          val deps = new Array[Int](nProducts)
          var i = 0
          while (i < nProducts) {
            deps(i) = rPartitioner.partitionIdFromBlockIndices(i, col)
            i += 1
          }
          deps
        }
      })

  private def block(hbm: HailBlockMatrix, bmPartitions: Array[Partition], p: HailGridPartitioner, context: TaskContext, i: Int, j: Int): BDM[Double] =
    try {
      hbm.blocks
        .iterator(bmPartitions(p.partitionIdFromBlockIndices(i, j)), context)
        .next()
        ._2
    } catch {
      case e: Exception => throw new RuntimeException(s"couldn't get block at $i, $j with partition id ${p.partitionIdFromBlockIndices(i,j)}; ${l.rows} ${l.cols} ${l.blockSize}; ${r.rows} ${r.cols} ${r.blockSize}; --- ; ${lPartitions.toSeq}, ${rPartitions.toSeq} ;; $nParts ;; $rowBlocks, $colBlocks", e)
    }

  private def leftBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(l, lPartitions, lPartitioner, context, i, j)
  private def rightBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(r, rPartitions, rPartitioner, context, i, j)

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val row = _partitioner.blockRowIndex(split.index)
    val col = _partitioner.blockColIndex(split.index)
    val rowsInThisBlock: Int = if (row + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else blockSize
    val colsInThisBlock: Int = if (col + 1 == colBlocks && colsRemainder != 0) colsRemainder else blockSize

    val result = BDM.zeros[Double](rowsInThisBlock, colsInThisBlock)
    var i = 0
    while (i < nProducts) {
      val left = leftBlock(row, i, context)
      val right = rightBlock(i, col, context)
      try {
        result :+= (left * right)
      } catch {
        case e: Exception => throw new RuntimeException(s"$row , $i , $col ;; $rowBlocks , $nProducts , $colBlocks ;; ${result.rows} x ${result.cols} :+= (${left.rows} x ${left.cols} * ${right.rows} x ${right.cols})", e)
      }
      i += 1
    }

    Iterator.single(((row, col), result))
  }

  protected def getPartitions: Array[Partition] =
    (0 until nParts).map(IntPartition.apply _).toArray[Partition]

  private val _partitioner = HailGridPartitioner(rows, cols, blockSize)
  /** Optionally overridden by subclasses to specify how they are partitioned. */
  @transient override val partitioner: Option[Partitioner] =
    Some(_partitioner)
}

case class IntPartition(index: Int) extends Partition { }
