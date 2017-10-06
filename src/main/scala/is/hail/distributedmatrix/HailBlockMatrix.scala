package is.hail.distributedmatrix

import java.io._

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail._
import is.hail.utils._
import org.apache.hadoop.io._
import org.apache.spark._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s._

object HailBlockMatrix {
  type M = HailBlockMatrix

  def from(sc: SparkContext, lm: BDM[Double], blockSize: Int): M = {
    val partitioner = HailGridPartitioner(lm.rows, lm.cols, blockSize)
    val lmBc = sc.broadcast(lm)
    val rowPartitions = partitioner.rowPartitions
    val colPartitions = partitioner.colPartitions
    val rowsRemainder = lm.rows % blockSize
    val colsRemainder = lm.cols % blockSize
    val indices = for {
      i <- 0 until rowPartitions
      j <- 0 until colPartitions
    } yield (i, j)
    val blocks = sc.parallelize(indices).map { case (i, j) =>
      val rowsInThisBlock = if (i + 1 == rowPartitions && rowsRemainder != 0) rowsRemainder else blockSize
      val colsInThisBlock = if (j + 1 == colPartitions && colsRemainder != 0) colsRemainder else blockSize
      val a = new Array[Double](rowsInThisBlock * colsInThisBlock)
      for {
        ii <- 0 until rowsInThisBlock
        jj <- 0 until colsInThisBlock
      } {
        a(jj * rowsInThisBlock + ii) = lmBc.value(i * blockSize + ii, j * blockSize + jj)
      }
      ((i, j), new BDM(rowsInThisBlock, colsInThisBlock, a))
    }.partitionBy(partitioner)
    new HailBlockMatrix(blocks, blockSize, lm.rows, lm.cols)
  }

  def from(irm: IndexedRowMatrix, blockSize: Int): M =
    irm.toHailBlockMatrixDense(blockSize)

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b, c, d, f)

  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r, f)

  private class PairWriter(var i: Int, var j: Int) extends Writable {
    def this() {
      this(0, 0)
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

  private class MatrixWriter(var rows: Int, var cols: Int, var a: Array[Double]) extends Writable {
    def this() {
      this(0, 0, null)
    }

    def write(out: DataOutput) {
      out.writeInt(rows)
      out.writeInt(cols)
      var i = 0
      while (i < rows * cols) {
        out.writeDouble(a(i))
        i += 1
      }
    }

    def readFields(in: DataInput) {
      rows = in.readInt()
      cols = in.readInt()
      a = new Array[Double](rows * cols)
      var i = 0
      while (i < rows * cols) {
        a(i) = in.readDouble()
        i += 1
      }
    }

    def toDenseMatrix(): BDM[Double] = {
      new BDM[Double](rows, cols, a)
    }
  }

  private val metadataRelativePath = "/metadata.json"
  private val matrixRelativePath = "/matrix"

  /**
    * Writes the matrix {@code m} to a Hadoop sequence file at location {@code
    * uri}.
    *
    **/
  def write(dm: M, uri: String) {
    val hadoop = dm.blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    dm.blocks.map { case ((i, j), lm) =>
      (new PairWriter(i, j), new MatrixWriter(lm.rows, lm.cols, lm.data))
    }
      .saveAsSequenceFile(uri + matrixRelativePath)

    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      jackson.Serialization.write(
        HailBlockMatrixMetadata(dm.blockSize, dm.rows, dm.cols),
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

    val blocks = hc.sc.sequenceFile[PairWriter, MatrixWriter](uri + matrixRelativePath).map { case (pw, mw) =>
      ((pw.i, pw.j), mw.toDenseMatrix())
    }

    val HailBlockMatrixMetadata(blockSize, rows, cols) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr =>
        jackson.Serialization.read[HailBlockMatrixMetadata](isr)
      }

    new HailBlockMatrix(blocks.partitionBy(HailGridPartitioner(rows, cols, blockSize)), blockSize, rows, cols)
  }

  object ops {
    implicit class Shim(l: M) {
      def t: M =
        l.transpose()
      def diag: Array[Double] =
        l.diagonal()

      def *(r: M): M =
        l.multiply(r)
      def *(r: BDM[Double]): M =
        l.multiply(r)

      def :+(r: M): M =
        l.add(r)
      def :-(r: M): M =
        l.subtract(r)
      def :*(r: M): M =
        l.pointwiseMultiply(r)
      def :/(r: M): M =
        l.pointwiseDivide(r)

      def +(r: Double): M =
        l.scalarAdd(r)
      def -(r: Double): M =
        l.scalarSubtract(r)
      def *(r: Double): M =
        l.scalarMultiply(r)
      def /(r: Double): M =
        l.scalarDivide(r)

      def :+(v: Array[Double]): M =
        l.vectorAddToEveryColumn(v)
      def :*(v: Array[Double]): M =
        l.vectorPointwiseMultiplyEveryColumn(v)

      def --*(v: Array[Double]): M =
        l.vectorPointwiseMultiplyEveryRow(v)
    }
    implicit class ScalarShim(l: Double) {
      def +(r: M): M =
        r.scalarAdd(l)
      def -(r: M): M = {
        val ll = l
        r.blockMap(ll - _)
      }
      def *(r: M): M =
        r.scalarMultiply(l)
      def /(r: M): M = {
        val ll = l
        r.blockMap(l / _)
      }
    }
  }
}

// must be top-level for Jackson to serialize correctly
case class HailBlockMatrixMetadata(blockSize: Int, rows: Long, cols: Long)

class HailBlockMatrix(val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val rows: Long,
  val cols: Long) extends Serializable {
  type M = HailBlockMatrix

  val st = Thread.currentThread().getStackTrace().mkString("\n")

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[HailGridPartitioner])

  val partitioner = blocks.partitioner.get.asInstanceOf[HailGridPartitioner]

  def transpose(): M =
    new HailBlockMatrix(new HailBlockMatrixTransposeRDD(this), blockSize, cols, rows)

  def diagonal(): Array[Double] = {
    require(rows == cols,
      s"diagonal only works on square matrices, given ${ rows }x${ cols }")

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

    blocks
      .filter { case ((i, j), block) => i == j }
      .map { case ((i, j), block) => (i, diagonal(block)) }
      .collect()
      .sortBy(_._1)
      .map(_._2)
      .fold(Array[Double]())(_ ++ _)
  }

  def add(that: M): M =
    blockMap2(that, _ + _)

  def subtract(that: M): M =
    blockMap2(that, _ - _)

  def pointwiseMultiply(that: M): M =
    blockMap2(that, _ :* _)

  def pointwiseDivide(that: M): M =
    blockMap2(that, _ :/ _)

  def multiply(that: M): M =
    new HailBlockMatrix(new HailBlockMatrixMultiplyRDD(this, that), blockSize, rows, that.cols)

  def multiply(lm: BDM[Double]): M = {
    require(cols == lm.rows,
      s"incompatible matrix dimensions: ${ rows }x${ cols } and ${ lm.rows }x${ lm.cols }")
    multiply(HailBlockMatrix.from(blocks.sparkContext, lm, blockSize))
  }

  def scalarAdd(i: Double): M =
    blockMap(_ + i)

  def scalarSubtract(i: Double): M =
    blockMap(_ - i)

  def scalarMultiply(i: Double): M =
    blockMap(_ :* i)

  def scalarDivide(i: Double): M =
    blockMap(_ / i)

  def vectorAddToEveryColumn(v: Array[Double]): M = {
    require(v.length == rows, s"vector length, ${ v.length }, must equal number of matrix rows ${ rows }; v: $v, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, j, x) => x + vBc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): M = {
    require(v.length == rows, s"vector length, ${ v.length }, must equal number of matrix rows ${ rows }; v: $v, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, j, x) => x * vBc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): M = {
    require(v.length == cols, s"vector length, ${ v.length }, must equal number of matrix columns ${ cols }; v: $v, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, j, x) => x * vBc.value(j.toInt))
  }

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
      s"Int.MaxValue. Currently numRows: ${ this.rows }")
    require(this.cols < Int.MaxValue, "The number of columns of this matrix should be less than " +
      s"Int.MaxValue. Currently numCols: ${ this.cols }")
    require(this.rows * this.cols < Int.MaxValue, "The length of the values array must be " +
      s"less than Int.MaxValue. Currently rows * cols: ${ this.rows * this.cols }")
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

  private def requireZippable(that: M) {
    require(rows == that.rows,
      s"must have same number of rows, but actually: ${ rows }x${ cols }, ${ that.rows }x${ that.cols }")
    require(cols == that.cols,
      s"must have same number of cols, but actually: ${ rows }x${ cols }, ${ that.rows }x${ that.cols }")
    require(blockSize == that.blockSize,
      s"blocks must be same size, but actually were ${ blockSize }x${ blockSize } and ${ that.blockSize }x${ that.blockSize }")
  }
  private def assertCompatibleLocalMatrix(lm: BDM[Double]) {
    assert(lm.offset == 0, s"${lm.offset}")
    assert(lm.majorStride == (if (lm.isTranspose) lm.cols else lm.rows), s"${lm.majorStride} ${lm.isTranspose} ${lm.rows} ${lm.cols}}")
  }

  def blockMap(op: BDM[Double] => BDM[Double]): M =
    new HailBlockMatrix(blocks.mapValues(op), blockSize, rows, cols)

  def blockMap2(that: M, op: (BDM[Double], BDM[Double]) => BDM[Double]): M = {
    requireZippable(that)
    new HailBlockMatrix(blocks.join(that.blocks).mapValues(op.tupled), blockSize, rows, cols)
  }

  def map(op: Double => Double): M = {
    val blocks = this.blocks.mapValues { lm =>
      assertCompatibleLocalMatrix(lm)
      val src = lm.data
      val dst = new Array[Double](src.length)
      var i = 0
      while (i < src.length) {
        dst(i) = op(src(i))
        i += 1
      }
      new BDM(lm.rows, lm.cols, dst, 0, lm.majorStride, lm.isTranspose)
    }
    new HailBlockMatrix(blocks, blockSize, rows, cols)
  }

  def map2(that: M, op: (Double, Double) => Double): M = {
    requireZippable(that)
    val blocks = this.blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = {
          assert(thisIter.hasNext == thatIter.hasNext)
          thisIter.hasNext
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((i1,j1), lm1) = thisIter.next()
          val ((i2,j2), lm2) = thatIter.next()
          assertCompatibleLocalMatrix(lm1)
          assertCompatibleLocalMatrix(lm2)
          assert(i1 == i2, s"$i1 $i2")
          assert(j1 == j2, s"$j1 $j2")
          val rows = lm1.rows
          val cols = lm1.cols
          val src1 = lm1.data
          val src2 = lm2.data
          val dst = new Array[Double](src1.length)
          var k = 0
          if (lm1.isTranspose == lm2.isTranspose) {
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k))
              k += 1
            }
          } else {
            while (k < src1.length) {
              val ii = k % lm1.majorStride
              val jj = k / lm1.majorStride
              val k2 = jj + ii * lm2.majorStride
              dst(k) = op(src1(k), src2(k2))
              k += 1
            }
          }
          ((i1,j1), new BDM(rows, cols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new HailBlockMatrix(blocks, blockSize, rows, cols)
  }

  def map4(dm2: M, dm3: M, dm4: M, op: (Double, Double, Double, Double) => Double): M = {
    requireZippable(dm2)
    requireZippable(dm3)
    requireZippable(dm4)
    val blocks = this.blocks.zipPartitions(dm2.blocks, dm3.blocks, dm4.blocks, preservesPartitioning = true) { (it1, it2, it3, it4) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = {
          assert(it1.hasNext == it2.hasNext)
          assert(it1.hasNext == it3.hasNext)
          assert(it1.hasNext == it4.hasNext)
          it1.hasNext
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((i1,j1), lm1) = it1.next()
          val ((i2,j2), lm2) = it2.next()
          val ((i3,j3), lm3) = it3.next()
          val ((i4,j4), lm4) = it4.next()
          assertCompatibleLocalMatrix(lm1)
          assertCompatibleLocalMatrix(lm2)
          assertCompatibleLocalMatrix(lm3)
          assertCompatibleLocalMatrix(lm4)
          assert(i1 == i2, s"$i1 $i2")
          assert(j1 == j2, s"$j1 $j2")
          assert(i1 == i3, s"$i1 $i3")
          assert(j1 == j3, s"$j1 $j3")
          assert(i1 == i4, s"$i1 $i4")
          assert(j1 == j4, s"$j1 $j4")
          val rows = lm1.rows
          val cols = lm1.cols
          val src1 = lm1.data
          val src2 = lm2.data
          val src3 = lm3.data
          val src4 = lm4.data
          val dst = new Array[Double](src1.length)
          var k = 0
          if (lm1.isTranspose == lm2.isTranspose
            && lm1.isTranspose == lm3.isTranspose
            && lm1.isTranspose == lm4.isTranspose) {
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k), src3(k), src4(k))
              k += 1
            }
          } else {
            // FIXME: code gen the optimal tree?
            // FIXME: code gen the optimal tree on driver?
            while (k < src1.length) {
              val ii = k % lm1.majorStride
              val jj = k / lm1.majorStride
              val v2 = if (lm1.isTranspose == lm2.isTranspose) src2(k) else src2(jj + ii * lm2.majorStride)
              val v3 = if (lm1.isTranspose == lm3.isTranspose) src3(k) else src3(jj + ii * lm3.majorStride)
              val v4 = if (lm1.isTranspose == lm4.isTranspose) src4(k) else src4(jj + ii * lm4.majorStride)
              dst(k) = op(src1(k), v2, v3, v4)
              k += 1
            }
          }
          ((i1, j1), new BDM(rows, cols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new HailBlockMatrix(blocks, blockSize, rows, cols)
  }

  def mapWithIndex(op: (Long, Long, Double) => Double): M = {
    val blockSize = this.blockSize
    val blocks = this.blocks.mapValuesWithKey { case ((i, j), lm) =>
      val iOffset = i.toLong * blockSize
      val jOffset = j.toLong * blockSize
      val size = lm.cols * lm.rows
      val result = new Array[Double](size)
      var jj = 0
      while (jj < lm.cols) {
        var ii = 0
        while (ii < lm.rows) {
          result(ii + jj * lm.rows) = op(iOffset + ii, jOffset + jj, lm(ii, jj))
          ii += 1
        }
        jj += 1
      }
      new BDM(lm.rows, lm.cols, result, 0, lm.rows, false)
    }
    new HailBlockMatrix(blocks, blockSize, rows, cols)
  }

  def map2WithIndex(that: M, op: (Long, Long, Double, Double) => Double): M = {
    requireZippable(that)
    val blockSize = this.blockSize
    val blocks = this.blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = {
          assert(thisIter.hasNext == thatIter.hasNext)
          thisIter.hasNext
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((i1,j1), lm1) = thisIter.next()
          val ((i2,j2), lm2) = thatIter.next()
          assertCompatibleLocalMatrix(lm1)
          assertCompatibleLocalMatrix(lm2)
          assert(i1 == i2, s"$i1 $i2")
          assert(j1 == j2, s"$j1 $j2")
          val iOffset = i.toLong * blockSize
          val jOffset = j.toLong * blockSize
          val size = lm1.cols * lm1.rows
          val result = new Array[Double](size)
          var jj = 0
          while (jj < lm1.cols) {
            var ii = 0
            while (ii < lm1.rows) {
              result(ii + jj * lm1.rows) = op(iOffset + ii, jOffset + jj, lm1(ii, jj), lm2(ii, jj))
              ii += 1
            }
            jj += 1
          }
          ((i, j), new BDM(lm1.rows, lm1.cols, result, 0, lm1.rows, false))
        }
      }
    }
    new HailBlockMatrix(blocks, blockSize, rows, cols)
  }

  def toIndexedRowMatrix(): IndexedRowMatrix = {
    require(cols <= Integer.MAX_VALUE)
    val icols = cols.toInt

    def seqOp(a: Array[Double], p: (Int, Array[Double])): Array[Double] = p match { case (offset, v) =>
      System.arraycopy(v, 0, a, offset, v.length)
      a
    }
    def combOp(l: Array[Double], r: Array[Double]): Array[Double] = {
      var i = 0
      while (i < l.length) {
        if (r(i) != 0)
          l(i) = r(i)
        i += 1
      }
      l
    }

    new IndexedRowMatrix(this.blocks.flatMap { case ((i, j), m) =>
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      for (k <- 0 until m.rows)
      yield (k + iOffset, (jOffset, m(k, ::).inner.toArray))
    }.aggregateByKey(new Array[Double](icols))(seqOp, combOp)
      .map { case (i, a) => new IndexedRow(i, new DenseVector(a)) },
      rows, icols)
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
    Some(prevPartitioner.transpose)
}

private class HailBlockMatrixMultiplyRDD(l: HailBlockMatrix, r: HailBlockMatrix)
  extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {
  require(l.cols == r.rows,
    s"inner dimension must match, but given: ${ l.rows }x${ l.cols }, ${ r.rows }x${ r.cols }")
  require(l.blockSize == r.blockSize,
    s"blocks must be same size, but actually were ${ l.blockSize }x${ l.blockSize } and ${ r.blockSize }x${ r.blockSize }")

  private val lPartitioner = l.partitioner
  private val lPartitions = l.blocks.partitions
  private val rPartitioner = r.partitioner
  private val rPartitions = r.blocks.partitions
  private val rows = l.rows
  private val cols = r.cols
  private val blockSize = l.blockSize
  private val rowPartitions = lPartitioner.rowPartitions
  private val colPartitions = rPartitioner.colPartitions
  private val nProducts = lPartitioner.colPartitions
  private val rowsRemainder = (rows % blockSize).toInt
  private val colsRemainder = (cols % blockSize).toInt
  private val nParts = rowPartitions * colPartitions

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

  private def block(dm: HailBlockMatrix, bmPartitions: Array[Partition], p: HailGridPartitioner, context: TaskContext, i: Int, j: Int): BDM[Double] = {
    val it = dm.blocks
      .iterator(bmPartitions(p.partitionIdFromBlockIndices(i, j)), context)
    assert(it.hasNext)
    val v = it.next()._2
    assert(!it.hasNext)
    v
  }

  private def leftBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(l, lPartitions, lPartitioner, context, i, j)

  private def rightBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(r, rPartitions, rPartitioner, context, i, j)

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val row = _partitioner.blockRowIndex(split.index)
    val col = _partitioner.blockColIndex(split.index)
    val rowsInThisBlock: Int = if (row + 1 == rowPartitions && rowsRemainder != 0) rowsRemainder else blockSize
    val colsInThisBlock: Int = if (col + 1 == colPartitions && colsRemainder != 0) colsRemainder else blockSize

    val result = BDM.zeros[Double](rowsInThisBlock, colsInThisBlock)
    var i = 0
    while (i < nProducts) {
      val left = leftBlock(row, i, context)
      val right = rightBlock(i, col, context)
      result :+= (left * right)
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

case class IntPartition(index: Int) extends Partition {}
