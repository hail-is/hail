package is.hail.distributedmatrix

import is.hail._
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import breeze.linalg.{DenseMatrix => BDM, Matrix => BM, _}
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.hadoop.io._
import java.io._
import org.json4s._
import java.net._

import scala.reflect.ClassTag

object HailBlockMatrixIsDistributedMatrix extends DistributedMatrix[HailBlockMatrix] {
  type M = HailBlockMatrix

  def cache(m: M): M = m.cache()

  def from(sc: SparkContext, bdm: BDM[Double], blockSize: Int): M = {
    val partitioner = HailGridPartitioner(bdm.rows, bdm.cols, blockSize)
    val rbc = sc.broadcast(bdm)
    val rowBlocks = partitioner.rowPartitions
    val colBlocks = partitioner.colPartitions
    val rowsRemainder = bdm.rows % blockSize
    val colsRemainder = bdm.cols % blockSize
    val indices = for {
      i <- 0 until rowBlocks
      j <- 0 until colBlocks
    } yield (i, j)
    val rMats = sc.parallelize(indices).map { case (i, j) =>
      val rowsInThisBlock = (if (i + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else blockSize)
      val colsInThisBlock = (if (j + 1 == colBlocks && colsRemainder != 0) colsRemainder else blockSize)
      val a = new Array[Double](rowsInThisBlock * colsInThisBlock)
      for {
        ii <- 0 until rowsInThisBlock
        jj <- 0 until colsInThisBlock
      } {
        a(jj*rowsInThisBlock + ii) = rbc.value(i*blockSize + ii, j*blockSize + jj)
      }
      ((i, j), new BDM(rowsInThisBlock, colsInThisBlock, a))
    }.partitionBy(partitioner)
    new HailBlockMatrix(rMats, blockSize, bdm.rows, bdm.cols)
  }
  def from(irm: IndexedRowMatrix): M =
    from(irm, true)
  def from(irm: IndexedRowMatrix, dense: Boolean): M =
    from(irm, dense, 1024)
  def from(irm: IndexedRowMatrix, blockSize: Int): M =
    from(irm, true, blockSize)
  def from(irm: IndexedRowMatrix, dense: Boolean, blockSize: Int): M =
    if (dense)
      irm.toHailBlockMatrixDense()
    else
      ???
  def from(bm: BlockMatrix): M = ???

  def transpose(m: M): M = m.transpose()
  def diagonal(m: M): Array[Double] = {
    require(m.rows == m.cols,
      s"diagonal only works on square matrices, given ${m.rows}x${m.cols}")

    def diagonal(block: BDM[Double]): Array[Double] = {
      val length = math.min(block.rows, block.cols)
      val diagonal = new Array[Double](length)
      var i = 0
      while (i < length) {
        diagonal(i) = block(i, i)
        i += 1
      }
      diagonal
    }

    m.blocks
      .filter{ case ((i, j), block) => i == j}
      .map { case ((i, j), block) => (i, diagonal(block)) }
      .collect()
      .sortBy(_._1)
      .map(_._2)
      .fold(Array[Double]())(_ ++ _)
  }

  def multiply(l: M, r: M): M = l.multiply(r)
  def multiply(l: M, r: BDM[Double]): M = {
    require(l.cols == r.rows,
      s"incompatible matrix dimensions: ${l.rows}x${l.cols} and ${r.rows}x${r.cols}")
    multiply(l, from(l.blocks.sparkContext, r, l.blockSize))
  }

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b,c,d,f)
  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r,f)
  def pointwiseAdd(l: M, r: M): M =
    l.add(r)
  def pointwiseSubtract(l: M, r: M): M =
    l.subtract(r)
  def pointwiseMultiply(l: M, r: M): M =
    l.pointwiseMultiply(r)
  def pointwiseDivide(l: M, r: M): M =
    l.pointwiseDivide(r)
  def map(f: Double => Double)(m: M): M =
    m.map(f)
  def scalarAdd(m: M, i: Double): M =
    m.blockMap(_ + i)
  def scalarSubtract(m: M, i: Double): M =
    m.blockMap(_ - i)
  def scalarSubtract(i: Double, m: M): M =
    m.blockMap(i - _)
  def scalarMultiply(m: M, i: Double): M =
    m.blockMap(_ :* i)
  def scalarDivide(m: M, i: Double): M =
    m.blockMap(_ / i)
  def scalarDivide(i: Double, m: M): M =
    m.blockMap(i / _)
  def vectorAddToEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.rows, s"vector length, ${v.length}, must equal number of matrix rows ${m.rows}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    m.mapWithIndex((i,j,x) => x + vbc.value(i.toInt))
  }
  def vectorPointwiseMultiplyEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.rows, s"vector length, ${v.length}, must equal number of matrix rows ${m.rows}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    m.mapWithIndex((i,j,x) => x * vbc.value(i.toInt))
  }
  def vectorPointwiseMultiplyEveryRow(v: Array[Double])(m: M): M = {
    require(v.length == m.cols, s"vector length, ${v.length}, must equal number of matrix columns ${m.cols}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    m.mapWithIndex((i,j,x) => x * vbc.value(j.toInt))
  }

  def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] = ???

  def toBlockRdd(m: M): RDD[((Int, Int), Matrix)] = ???

  def toLocalMatrix(m: M): BDM[Double] = m.toLocalMatrix()

  private class PairWriter(var i: Int, var j: Int) extends Writable {
    def this() {
      this(0,0)
    }

    def write(out: DataOutput) {
      out.writeInt(i)
      out.writeInt(j)
    }

    def readFields(in: DataInput) {
      i = in.readInt()
      j = in.readInt()
    }
  }

  private class MatrixWriter(var rows: Int, var cols: Int, var m: Array[Double]) extends Writable {
    def this() {
      this(0, 0, null)
    }

    def write(out: DataOutput) {
      out.writeInt(rows)
      out.writeInt(cols)
      var i = 0
      while (i < rows * cols) {
        out.writeDouble(m(i))
        i += 1
      }
    }

    def readFields(in: DataInput) {
      rows = in.readInt()
      cols = in.readInt()
      m = new Array[Double](rows * cols)
      var i = 0
      while (i < rows * cols) {
        m(i) = in.readDouble()
        i += 1
      }
    }

    def toDenseMatrix(): DenseMatrix = {
      new DenseMatrix(rows, cols, m)
    }

    def toBDM(): BDM[Double] = {
      new BDM[Double](rows, cols, m)
    }
  }

  private val metadataRelativePath = "/metadata.json"
  private val matrixRelativePath = "/matrix"
  /**
    * Writes the matrix {@code m} to a Hadoop sequence file at location {@code
    * uri}.
    *
    **/
  def write(m: M, uri: String) {
    val hadoop = m.blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    m.blocks.map { case ((i, j), m) =>
      (new PairWriter(i, j), new MatrixWriter(m.rows, m.cols, m.data)) }
      .saveAsSequenceFile(uri+matrixRelativePath)

    hadoop.writeDataFile(uri+metadataRelativePath) { os =>
      jackson.Serialization.write(
        HailBlockMatrixMetadata(m.blockSize, m.rows, m.cols),
        os)
    }
  }

  /**
    * Reads a BlockMatrix matrix written by {@code write} at location {@code
    * uri}.
    *
    **/
  def read(hc: HailContext, uri: String): M = {
    val hadoop = hc.hadoopConf
    hadoop.mkDir(uri)

    val rdd = hc.sc.sequenceFile[PairWriter, MatrixWriter](uri+matrixRelativePath).map { case (pw, mw) =>
      ((pw.i, pw.j), mw.toBDM())
    }

    val HailBlockMatrixMetadata(blockSize, rows, cols) =
      hadoop.readTextFile(uri+metadataRelativePath) { isr  =>
        jackson.Serialization.read[HailBlockMatrixMetadata](isr)
      }

    new HailBlockMatrix(rdd.partitionBy(HailGridPartitioner(rows, cols, blockSize)), blockSize, rows, cols)
  }
}

// must be top-level for Jackson to serialize correctly
case class HailBlockMatrixMetadata(blockSize: Int, rows: Long, cols: Long)

