package is.hail.distributedmatrix

import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.reflect.ClassTag

object BlockMatrixIsDistributedMatrix extends DistributedMatrix[BlockMatrix] {
  type M = BlockMatrix

  def cache(m: M): M = m.cache()

  def from(irm: IndexedRowMatrix, dense: Boolean = true): M =
    if (dense) irm.toBlockMatrixDense()
    else irm.toBlockMatrix()
  def from(bm: BlockMatrix): M = bm
  def from(cm: CoordinateMatrix): M =
    cm.toBlockMatrix()

  def transpose(m: M): M = m.transpose
  def diagonal(m: M): Array[Double] = {
    val rowsPerBlock = m.rowsPerBlock
    val colsPerBlock = m.colsPerBlock

    // check if the intervals [i, i+rows) and [j, j+rows) overlap,
    // meaning this block contains the diagonal.
    def containsDiagonal(coord: (Int, Int)): Boolean = coord match {
      case (blocki, blockj) =>
        val i = blocki * rowsPerBlock
        val j = blockj * colsPerBlock
        val i2 = i + rowsPerBlock
        val j2 = i + colsPerBlock

        i <= j && j < i2 ||
        i < j2 && j2 < i2 ||
        j <= i && i < j2 ||
        j < i2 && i2 < j2
    }

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
      .filter(x => containsDiagonal(x._1))
      .map { case ((i, j), m) => (math.max(i,j), diagonal(m)) }
      .reduce { case ((i, m1), (j, m2)) =>
        if (i < j) (i, m1 ++ m2)
        else (j, m2 ++ m1)
    }._2
  }

  def multiply(l: M, r: M): M = l.multiply(r)
  def multiply(l: M, r: DenseMatrix): M = {
    require(l.numCols() == r.numRows,
      s"incompatible matrix dimensions: ${l.numRows()}x${l.numCols()} and ${r.numRows}x${r.numCols}")
    val sc = l.blocks.sparkContext
    val rbc = sc.broadcast(r)
    val rRowsPerBlock = l.colsPerBlock
    val rColsPerBlock = l.rowsPerBlock
    val rRowBlocks = (r.numRows - 1) / rRowsPerBlock + 1
    val rColBlocks = (r.numCols - 1) / rColsPerBlock + 1
    val rRowsRemainder = r.numRows % rRowsPerBlock
    val rColsRemainder = r.numCols % rColsPerBlock
    val indices = for {
      i <- 0 until rRowBlocks
      j <- 0 until rColBlocks
    } yield (i, j)
    val rMats = sc.parallelize(indices).map { case (i, j) =>
      val rRowsInThisBlock = (if (i + 1 == rRowBlocks && rRowsRemainder != 0) rRowsRemainder else rRowsPerBlock)
      val rColsInThisBlock = (if (i + 1 == rColBlocks && rColsRemainder != 0) rColsRemainder else rColsPerBlock)
      val a = new Array[Double](rRowsInThisBlock * rColsInThisBlock)
      for {
        ii <- 0 until rRowsInThisBlock
        jj <- 0 until rColsInThisBlock
      } {
        a(jj*rRowsInThisBlock + ii) = rbc.value(i*rRowsPerBlock + ii, j*rColsPerBlock + jj)
      }
      ((i, j), new DenseMatrix(rRowsInThisBlock, rColsInThisBlock, a, false): Matrix)
    }.partitionBy(GridPartitioner(rRowBlocks, rColBlocks))
    l.multiply(new BlockMatrix(rMats, rRowsPerBlock, rColsPerBlock, r.numRows, r.numCols))
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
    require(l.rowsPerBlock == r.rowsPerBlock)
    require(l.colsPerBlock == r.colsPerBlock)
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
  def scalarMultiply(m: M, i: Double): M = map(_ * i)(m)
  def scalarDivide(m: M, i: Double): M = map(_ / i)(m)
  def scalarAdd(i: Double, m: M): M = map(i + _)(m)
  def scalarSubtract(i: Double, m: M): M = map(i - _)(m)
  def scalarMultiply(i: Double, m: M): M = map(i * _)(m)
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
      new DenseMatrix(m.numRows, m.numCols, m.toArray.zipWithIndex.map { case (e, j) =>
        val rowsInThisBlock: Int = (if (blockRow + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
        val colsInThisBlock: Int = (if (blockCol + 1 == colBlocks && colsRemainder != 0) colsRemainder else colsPerBlock)
        if (blockRow.toLong * rowsInThisBlock + j % rowsInThisBlock < nRows &&
          blockCol.toLong * colsInThisBlock + j / rowsInThisBlock < nCols)
          op(e, blockRow * rowsInThisBlock + j % rowsInThisBlock)
        else
          e
      })
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
      new DenseMatrix(m.numRows, m.numCols, m.toArray.zipWithIndex.map { case (e, j) =>
        val rowsInThisBlock: Int = (if (blockRow + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else rowsPerBlock)
        val colsInThisBlock: Int = (if (blockCol + 1 == colBlocks && colsRemainder != 0) colsRemainder else colsPerBlock)
        if (blockRow * rowsInThisBlock + j % rowsInThisBlock < nRows &&
          blockCol * colsInThisBlock + j / rowsInThisBlock < nCols)
          op(e, blockCol * colsInThisBlock + j / rowsInThisBlock)
        else
          e
      })
    }
    new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, nRows, nCols)
  }

  def vectorAddToEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.numRows())
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithRowIndex((x,i) => x + vbc.value(i))(m)
  }
  def vectorPointwiseMultiplyEveryColumn(v: Array[Double])(m: M): M = {
    require(v.length == m.numRows())
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithRowIndex((x,i) => x * vbc.value(i))(m)
  }
  def vectorPointwiseMultiplyEveryRow(v: Array[Double])(m: M): M = {
    require(v.length == m.numCols())
    val vbc = m.blocks.sparkContext.broadcast(v)
    mapWithColIndex((x,i) => x * vbc.value(i))(m)
  }

  def mapRows[U](m: M, f: Array[Double] => U)(implicit uct: ClassTag[U]): RDD[U] =
    m.toIndexedRowMatrix().rows.map((ir: IndexedRow) => f(ir.vector.toArray))

  def toBlockRdd(m: M): RDD[((Int, Int), Matrix)] = m.blocks
  def toCoordinateMatrix(m: M): CoordinateMatrix = m.toCoordinateMatrix()

  def toLocalMatrix(m: M): Matrix = m.toLocalMatrix()
}
