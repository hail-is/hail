package is.hail.distributedmatrix

import is.hail._
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.hadoop.io._
import java.io._

import org.json4s._

import org.apache.commons.lang3.StringUtils

import scala.reflect.ClassTag

object BlockMatrixIsDistributedMatrix extends DistributedMatrix[BlockMatrix] {
  type M = BlockMatrix

  def cache(m: M): M = m.cache()

  def from(irm: IndexedRowMatrix, dense: Boolean = true): M =
    if (dense) irm.toBlockMatrixDense()
    else irm.toBlockMatrix()
  def from(sc: SparkContext, dm: DenseMatrix, rowsPerBlock: Int, colsPerBlock: Int): M = {
    val rbc = sc.broadcast(dm)
    val rowBlocks = (dm.numRows - 1) / rowsPerBlock + 1
    val colBlocks = (dm.numCols - 1) / colsPerBlock + 1
    val rowsRemainder = dm.numRows % rowsPerBlock
    val colsRemainder = dm.numCols % colsPerBlock
    val indices = for {
      i <- 0 until rowBlocks
      j <- 0 until colBlocks
    } yield (i, j)
    val rMats = sc.parallelize(indices).map { case (i, j) =>
      val rowsInThisBlock = (if (i + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
      val colsInThisBlock = (if (j + 1 == colBlocks && colsRemainder != 0) colsRemainder else colsPerBlock)
      val a = new Array[Double](rowsInThisBlock * colsInThisBlock)
      for {
        ii <- 0 until rowsInThisBlock
        jj <- 0 until colsInThisBlock
      } {
        a(jj*rowsInThisBlock + ii) = rbc.value(i*rowsPerBlock + ii, j*colsPerBlock + jj)
      }
      ((i, j), new DenseMatrix(rowsInThisBlock, colsInThisBlock, a, false): Matrix)
    }.partitionBy(GridPartitioner(rowBlocks, colBlocks))
    new BlockMatrix(rMats, rowsPerBlock, colsPerBlock, dm.numRows, dm.numCols)
  }
  def from(irm: IndexedRowMatrix, rowsPerBlock: Int, colsPerBlock: Int): M =
    irm.toBlockMatrixDense(rowsPerBlock, colsPerBlock)
  def from(bm: BlockMatrix): M =
    new BlockMatrix(
      bm.blocks.partitionBy(GridPartitioner(((bm.numRows - 1) / bm.rowsPerBlock).toInt + 1, ((bm.numCols - 1) / bm.colsPerBlock).toInt + 1)).persist(),
      bm.rowsPerBlock, bm.colsPerBlock, bm.numRows(), bm.numCols())
  def from(bm: BlockMatrix, rowsPerBlock: Int, colsPerBlock: Int): M =
    new BlockMatrix(
      bm.blocks.partitionBy(GridPartitioner(((bm.numRows - 1) / rowsPerBlock).toInt + 1, ((bm.numCols - 1) / colsPerBlock).toInt + 1)).persist(),
      rowsPerBlock, colsPerBlock, bm.numRows(), bm.numCols())

  def transpose(m: M): M =
    BetterBlockMatrix.transpose(m)
  def diagonal(m: M): Array[Double] = {
    val rowsPerBlock = m.rowsPerBlock
    val colsPerBlock = m.colsPerBlock

    // FIXME: generalize to arbitrary block sizes
    require(rowsPerBlock == colsPerBlock)

    def diagonal(block: Matrix): Array[Double] = {
      val length = math.min(block.numRows,block.numCols)
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
      .map { case ((i, j), m) => (i, diagonal(m)) }
      .collect()
      .sortBy(_._1)
      .map(_._2)
      .fold(Array[Double]())(_ ++ _)
  }

  /**
    * Note that this method requires that the individual blocks of l have the same number of columns as the
    * blocks of r have rows.
    */
  def multiply(l: M, r: M): M =
    BetterBlockMatrix.multiply(l, r)
  def multiply(l: M, r: DenseMatrix): M = {
    require(l.numCols() == r.numRows,
      s"incompatible matrix dimensions: ${l.numRows()}x${l.numCols()} and ${r.numRows}x${r.numCols}")
    multiply(l, from(l.blocks.sparkContext, r, l.colsPerBlock, l.rowsPerBlock))
  }

  def map4(op: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M = {
    require(a.numRows() == b.numRows(), s"expected a's dimensions to match b's dimensions, but: ${a.numRows()} x ${a.numCols()},  ${b.numRows()} x ${b.numCols()}")
    require(b.numRows() == c.numRows(), s"expected b's dimensions to match c's dimensions, but: ${b.numRows()} x ${b.numCols()},  ${c.numRows()} x ${c.numCols()}")
    require(c.numRows() == d.numRows(), s"expected c's dimensions to match d's dimensions, but: ${c.numRows()} x ${c.numCols()},  ${d.numRows()} x ${d.numCols()}")
    require(a.numCols() == b.numCols())
    require(b.numCols() == c.numCols())
    require(c.numCols() == d.numCols())
    require(a.rowsPerBlock == b.rowsPerBlock)
    require(b.rowsPerBlock == c.rowsPerBlock)
    require(c.rowsPerBlock == d.rowsPerBlock)
    require(a.colsPerBlock == b.colsPerBlock)
    require(b.colsPerBlock == c.colsPerBlock)
    require(c.colsPerBlock == d.colsPerBlock)
    val blocks: RDD[((Int, Int), Matrix)] = a.blocks.join(b.blocks).join(c.blocks).join(d.blocks).mapValues { case (((m1, m2), m3), m4) =>
      val size = m1.numRows * m1.numCols
      val result = new Array[Double](size)
      var j = 0
      while (j < m1.numCols) {
        var i = 0
        while (i < m1.numRows) {
          result(i + j*m1.numRows) = op(m1(i, j), m2(i, j), m3(i, j), m4(i, j))
          i += 1
        }
        j += 1
      }
      new DenseMatrix(m1.numRows, m1.numCols, result)
    }
    new BlockMatrix(blocks, a.rowsPerBlock, a.colsPerBlock, a.numRows(), a.numCols())
  }

  def map2(op: (Double, Double) => Double)(l: M, r: M): M = {
    require(l.numRows() == r.numRows())
    require(l.numCols() == r.numCols())
    require(l.rowsPerBlock == r.rowsPerBlock, s"blocks must be same size, but actually were ${l.rowsPerBlock}x${l.colsPerBlock} and ${r.rowsPerBlock}x${l.colsPerBlock}")
    require(l.colsPerBlock == r.colsPerBlock, s"blocks must be same size, but actually were ${l.rowsPerBlock}x${l.colsPerBlock} and ${r.rowsPerBlock}x${l.colsPerBlock}")
    val blocks: RDD[((Int, Int), Matrix)] = l.blocks.join(r.blocks).mapValues { case (m1, m2) =>
      val size = m1.numRows * m1.numCols
      val result = new Array[Double](size)
      var j = 0
      while (j < m1.numCols) {
        var i = 0
        while (i < m1.numRows) {
          result(i + j*m1.numRows) = op(m1(i, j), m2(i, j))
          i += 1
        }
        j += 1
      }
      new DenseMatrix(m1.numRows, m1.numCols, result)
    }
    new BlockMatrix(blocks, l.rowsPerBlock, l.colsPerBlock, l.numRows(), l.numCols())
  }

  def map2WithIndex(op: (Long, Long, Double, Double) => Double)(l: M, r: M): M = {
    require(l.numRows() == r.numRows())
    require(l.numCols() == r.numCols())
    require(l.rowsPerBlock == r.rowsPerBlock, s"blocks must be same size, but actually were ${l.rowsPerBlock}x${l.colsPerBlock} and ${r.rowsPerBlock}x${l.colsPerBlock}")
    require(l.colsPerBlock == r.colsPerBlock, s"blocks must be same size, but actually were ${l.rowsPerBlock}x${l.colsPerBlock} and ${r.rowsPerBlock}x${l.colsPerBlock}")
    val rowsPerBlock = l.rowsPerBlock
    val colsPerBlock = l.colsPerBlock
    val blocks: RDD[((Int, Int), Matrix)] = l.blocks.join(r.blocks).mapValuesWithKey { case ((blocki, blockj), (m1, m2)) =>
      val iprefix = blocki.toLong * rowsPerBlock
      val jprefix = blockj.toLong * colsPerBlock
      val size = m1.numRows * m1.numCols
      val result = new Array[Double](size)
      var j = 0
      while (j < m1.numCols) {
        var i = 0
        while (i < m1.numRows) {
          result(i + j*m1.numRows) = op(iprefix + i, jprefix + j, m1(i, j), m2(i, j))
          i += 1
        }
        j += 1
      }
      new DenseMatrix(m1.numRows, m1.numCols, result)
    }
    new BlockMatrix(blocks, l.rowsPerBlock, l.colsPerBlock, l.numRows(), l.numCols())
  }

  def pointwiseAdd(l: M, r: M): M = map2(_ + _)(l, r)
  def pointwiseSubtract(l: M, r: M): M = map2(_ - _)(l, r)
  def pointwiseMultiply(l: M, r: M): M = map2(_ * _)(l, r)
  def pointwiseDivide(l: M, r: M): M = map2(_ / _)(l, r)

  def map(op: Double => Double)(m: M): M = {
    val blocks: RDD[((Int, Int), Matrix)] = m.blocks.mapValues { case m =>
      val size = m.numRows * m.numCols
      val result = new Array[Double](size)
      var j = 0
      while (j < m.numCols) {
        var i = 0
        while (i < m.numRows) {
          result(i + j*m.numRows) = op(m(i, j))
          i += 1
        }
        j += 1
      }
      new DenseMatrix(m.numRows, m.numCols, result)
    }
    new BlockMatrix(blocks, m.rowsPerBlock, m.colsPerBlock, m.numRows(), m.numCols())
  }
  def scalarAdd(m: M, i: Double): M = map(_ + i)(m)
  def scalarSubtract(m: M, i: Double): M = map(_ - i)(m)
  def scalarSubtract(i: Double, m: M): M = map(i - _)(m)
  def scalarMultiply(m: M, i: Double): M = map(_ * i)(m)
  def scalarDivide(m: M, i: Double): M = map(_ / i)(m)
  def scalarDivide(i: Double, m: M): M = map(i / _)(m)

  private def mapWithRowIndex(op: (Double, Int) => Double)(x: M): M = {
    val nRows: Long = x.numRows
    val nCols: Long = x.numCols
    val rowsPerBlock: Int = x.rowsPerBlock
    val colsPerBlock: Int = x.colsPerBlock
    val rowBlocks: Int = ((nRows - 1) / rowsPerBlock).toInt + 1
    val colBlocks: Int = ((nCols - 1) / colsPerBlock).toInt + 1
    val rowsRemainder: Int = (nRows % rowsPerBlock).toInt
    val colsRemainder: Int = (nCols % colsPerBlock).toInt
    val blocks: RDD[((Int, Int), Matrix)] = x.blocks.mapValuesWithKey { case ((blockRow, blockCol), m) =>
      val rowsInThisBlock: Int = (if (blockRow + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
      val a = m.toArray
      var i = 0
      while (i < a.length) {
        assert(blockRow.toLong * rowsPerBlock + i % rowsInThisBlock < nRows &&
          blockCol.toLong * colsPerBlock + i / rowsInThisBlock < nCols)

        a(i) = op(a(i), blockRow * rowsPerBlock + i % rowsInThisBlock)
        i += 1
      }
      new DenseMatrix(m.numRows, m.numCols, a)
    }
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, nRows, nCols)
  }
  private def mapWithColIndex(op: (Double, Int) => Double)(x: M): M = {
    val nRows = x.numRows
    val nCols = x.numCols
    val rowsPerBlock = x.rowsPerBlock
    val colsPerBlock = x.colsPerBlock
    val rowBlocks = ((nRows - 1) / rowsPerBlock).toInt + 1
    val colBlocks = ((nCols - 1) / colsPerBlock).toInt + 1
    val rowsRemainder = (nRows % rowsPerBlock).toInt
    val colsRemainder = (nCols % colsPerBlock).toInt
    val blocks: RDD[((Int, Int), Matrix)] = x.blocks.mapValuesWithKey { case ((blockRow, blockCol), m) =>
      val rowsInThisBlock: Int = (if (blockRow + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
      val a = m.toArray
      var i = 0
      while (i < a.length) {
        assert(blockRow * rowsPerBlock + i % rowsInThisBlock < nRows &&
          blockCol * colsPerBlock + i / rowsInThisBlock < nCols)

        a(i) = op(a(i), blockCol * colsPerBlock + i / rowsInThisBlock)
        i += 1
      }
      new DenseMatrix(m.numRows, m.numCols, a)
    }
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, nRows, nCols)
  }

  def vectorAddToEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.numRows(), s"vector length, ${v.length}, must equal number of matrix rows ${m.numRows()}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithRowIndex((x,i) => x + vbc.value(i))(m)
  }
  def vectorPointwiseMultiplyEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.numRows(), s"vector length, ${v.length}, must equal number of matrix rows ${m.numRows()}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithRowIndex((x,i) => x * vbc.value(i))(m)
  }
  def vectorPointwiseMultiplyEveryRow(v: Array[Double])(m: M): M = {
    require(v.length == m.numCols(), s"vector length, ${v.length}, must equal number of matrix columns ${m.numCols()}; v: $v, m: $m")
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithColIndex((x,i) => x * vbc.value(i))(m)
  }

  def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] =
    m.toIndexedRowMatrix().rows.map((ir: IndexedRow) => f(ir.vector.toArray))

  def toBlockRdd(m: M): RDD[((Int, Int), Matrix)] = m.blocks

  def toLocalMatrix(m: M): Matrix = m.toLocalMatrix()

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

    writeBlocks(m, uri)
    
    hadoop.writeDataFile(uri+metadataRelativePath) { os =>
      jackson.Serialization.write(
        BlockMatrixMetadata(m.rowsPerBlock, m.colsPerBlock, m.numRows(), m.numCols()),
        os)
    }
  }
  
  def writeBlocks(m: M, uri: String) {
    val blocks = m.blocks
    val sc = blocks.sparkContext
    val hadoopConf = sc.hadoopConfiguration
    
    hadoopConf.mkDir(uri + "/blockstore")
    
    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))
    
    val nPartitions = blocks.getNumPartitions
    val d = digitsNeeded(nPartitions)

    val blockCount = blocks.mapPartitionsWithIndex { case (i, it) =>
      var localBlockCount = 0L

      val is = i.toString
      assert(is.length <= d)
      val pis = StringUtils.leftPad(is, d, "0")

      sHadoopConfBc.value.value.writeDataFile(uri + "/blockstore/block-" + pis) { out =>
        it.foreach { case ((iblock, jblock), dm) => // will have one block per partition

          new MatrixWriter(dm.numRows, dm.numCols, dm.toArray).write(out)
          
          localBlockCount += 1
        }
      }

      Iterator(localBlockCount)
    }
      .fold(0L)(_ + _)

    info(s"wrote $blockCount blocks") // should equal number of partitions
  }

  /**
    * Reads a BlockMatrix matrix written by {@code write} at location {@code
    * uri}.
    *
    **/
  def read(hc: HailContext, uri: String): M = {
    val hadoop = hc.hadoopConf
    hadoop.mkDir(uri)
    
    val BlockMatrixMetadata(rowsPerBlock, colsPerBlock, nRowBlocks, nColBlocks) =
      hadoop.readTextFile(uri+metadataRelativePath) { isr  =>
        jackson.Serialization.read[BlockMatrixMetadata](isr)
      }
    
    val blocks = new ReadBlocksRDD(hc.sc, uri, nRowBlocks, nColBlocks)
    
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, nRowBlocks, nColBlocks)
  }
}

case class ReadBlocksRDDPartition(index: Int) extends Partition

class ReadBlocksRDD(sc: SparkContext, uri: String, nRowBlocks: Long, nColBlocks: Long) extends RDD[((Int, Int), Matrix)](sc, Nil) {
  assert(nRowBlocks < Int.MaxValue && nColBlocks < Int.MaxValue)
  private val nBlocks = nRowBlocks * nColBlocks
  
  assert(nBlocks < Int.MaxValue)
  val nPartitions: Int = nBlocks.toInt
  
  override def getPartitions: Array[Partition] =
    Array.tabulate(nPartitions)(i => ReadBlocksRDDPartition(i))
  
  private val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  override def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), Matrix)] = {
    val d = digitsNeeded(nPartitions)

    val is = split.index.toString
    assert(is.length <= d)
    val pis = StringUtils.leftPad(is, d, "0")

    Iterator(sHadoopConfBc.value.value.readDataFile(uri + "/rowstore/part-" + pis) { in =>
      val mw = new MatrixWriter()
      mw.readFields(in)

      ((mw.rows, mw.cols), mw.toDenseMatrix())
    })
  }

  override val partitioner: Option[Partitioner] = Some(GridPartitioner(nRowBlocks.toInt, nColBlocks.toInt))
}

// must be top-level for Jackson to serialize correctly
case class BlockMatrixMetadata(rowsPerBlock: Int, colsPerBlock: Int, numRows: Long, numCols: Long)

class PairWriter(var i: Int, var j: Int) extends Writable {
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

class MatrixWriter(var rows: Int, var cols: Int, var m: Array[Double]) extends Writable {
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
}