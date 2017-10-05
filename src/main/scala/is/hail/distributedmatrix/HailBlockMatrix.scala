package is.hail.distributedmatrix

import java.io._

import breeze.linalg.{DenseMatrix => BDM, Matrix => BM, _}
import is.hail._
import is.hail.utils._
import org.apache.hadoop.io._
import org.apache.spark.{SparkContext, _}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s._

import scala.reflect.ClassTag

object HailBlockMatrix {
  type M = HailBlockMatrix

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
        a(jj * rowsInThisBlock + ii) = rbc.value(i * blockSize + ii, j * blockSize + jj)
      }
      ((i, j), new BDM(rowsInThisBlock, colsInThisBlock, a))
    }.partitionBy(partitioner)
    new HailBlockMatrix(rMats, blockSize, bdm.rows, bdm.cols)
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

    def toDenseMatrix(): BDM[Double] = {
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
      (new PairWriter(i, j), new MatrixWriter(m.rows, m.cols, m.data))
    }
      .saveAsSequenceFile(uri + matrixRelativePath)

    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
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

    val rdd = hc.sc.sequenceFile[PairWriter, MatrixWriter](uri + matrixRelativePath).map { case (pw, mw) =>
      ((pw.i, pw.j), mw.toDenseMatrix())
    }

    val HailBlockMatrixMetadata(blockSize, rows, cols) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr =>
        jackson.Serialization.read[HailBlockMatrixMetadata](isr)
      }

    new HailBlockMatrix(rdd.partitionBy(HailGridPartitioner(rows, cols, blockSize)), blockSize, rows, cols)
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
    require(this.rows == this.cols,
      s"diagonal only works on square matrices, given ${ this.rows }x${ this.cols }")

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

    this.blocks
      .filter { case ((i, j), block) => i == j }
      .map { case ((i, j), block) => (i, diagonal(block)) }
      .collect()
      .sortBy(_._1)
      .map(_._2)
      .fold(Array[Double]())(_ ++ _)
  }

  def add(other: M): M =
    blockMap2(other, _ + _)

  def subtract(other: M): M =
    blockMap2(other, _ - _)

  def pointwiseMultiply(other: M): M =
    blockMap2(other, _ :* _)

  def pointwiseDivide(other: M): M =
    blockMap2(other, _ :/ _)

  def multiply(other: M): M =
    new HailBlockMatrix(new HailBlockMatrixMultiplyRDD(this, other), blockSize, rows, other.cols)

  def multiply(r: BDM[Double]): M = {
    require(this.cols == r.rows,
      s"incompatible matrix dimensions: ${ this.rows }x${ this.cols } and ${ r.rows }x${ r.cols }")
    multiply(HailBlockMatrix.from(this.blocks.sparkContext, r, this.blockSize))
  }

  def scalarAdd(i: Double): M =
    this.blockMap(_ + i)

  def scalarSubtract(i: Double): M =
    this.blockMap(_ - i)

  def scalarMultiply(i: Double): M =
    this.blockMap(_ :* i)

  def scalarDivide(i: Double): M =
    this.blockMap(_ / i)

  def vectorAddToEveryColumn(v: Array[Double]): M = {
    require(v.length == this.rows, s"vector length, ${ v.length }, must equal number of matrix rows ${ this.rows }; v: $v, m: $this")
    val vbc = this.blocks.sparkContext.broadcast(v)
    this.mapWithIndex((i, j, x) => x + vbc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): M = {
    require(v.length == this.rows, s"vector length, ${ v.length }, must equal number of matrix rows ${ this.rows }; v: $v, m: $this")
    val vbc = this.blocks.sparkContext.broadcast(v)
    this.mapWithIndex((i, j, x) => x * vbc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): M = {
    require(v.length == this.cols, s"vector length, ${ v.length }, must equal number of matrix columns ${ this.cols }; v: $v, m: $this")
    val vbc = this.blocks.sparkContext.broadcast(v)
    this.mapWithIndex((i, j, x) => x * vbc.value(j.toInt))
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

  private def requireZippable(other: M) {
    require(rows == other.rows,
      s"must have same number of rows, but actually: ${ rows }x${ cols }, ${ other.rows }x${ other.cols }")
    require(cols == other.cols,
      s"must have same number of cols, but actually: ${ rows }x${ cols }, ${ other.rows }x${ other.cols }")
    require(blockSize == other.blockSize,
      s"blocks must be same size, but actually were ${ blockSize }x${ blockSize } and ${ other.blockSize }x${ other.blockSize }")
  }

  def blockMap(op: BDM[Double] => BDM[Double]): M =
    new HailBlockMatrix(blocks.mapValues(op), blockSize, rows, cols)

  def blockMap2(other: M, op: (BDM[Double], BDM[Double]) => BDM[Double]): M = {
    requireZippable(other)
    new HailBlockMatrix(blocks.join(other.blocks).mapValues(op.tupled), blockSize, rows, cols)
  }

  def map(op: Double => Double): M = {
    val blocks2 = blocks.mapValues { m =>
      assert(m.offset == 0, s"${m.offset}")
      assert(m.majorStride == (if (m.isTranspose) m.cols else m.rows), s"${m.majorStride} ${m.isTranspose} ${m.rows} ${m.cols}}")
      val src = m.data
      val dst = new Array[Double](src.length)
      var i = 0
      while (i < src.length) {
        dst(i) = op(src(i))
        i += 1
      }
      new BDM(m.rows, m.cols, dst, 0, m.majorStride, m.isTranspose)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map2(other: M, op: (Double, Double) => Double): M = {
    requireZippable(other)
    val blocks2 = blocks.zipPartitions(other.blocks, preservesPartitioning = true) { (thisIter, otherIter) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = (thisIter.hasNext, otherIter.hasNext) match {
          case (true, true) => true
          case (false, false) => false
          case _ => throw new RuntimeException("Can only zip RDDs with " +
              "same number of elements in each partition")
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((i,j), m1) = thisIter.next()
          val ((i2,j2), m2) = otherIter.next()
          assert(m1.offset == 0, s"${m1.offset}")
          assert(m1.majorStride == (if (m1.isTranspose) m1.cols else m1.rows), s"${m1.majorStride} ${m1.isTranspose} ${m1.rows} ${m1.cols}")
          assert(m2.offset == 0, s"${m2.offset}")
          assert(m2.majorStride == (if (m2.isTranspose) m2.cols else m2.rows), s"${m2.majorStride} ${m2.isTranspose} ${m2.rows} ${m2.cols}")
          assert(i == i2, s"$i $i2")
          assert(j == j2, s"$j $j2")
          val rows = m1.rows
          val cols = m1.cols
          val src1 = m1.data
          val src2 = m2.data
          val dst = new Array[Double](src1.length)
          var k = 0
          if (m1.isTranspose == m2.isTranspose) {
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k))
              k += 1
            }
          } else {
            while (k < src1.length) {
              val ii = k % m1.majorStride
              val jj = k / m1.majorStride
              val k2 = jj + ii * m2.majorStride
              dst(k) = op(src1(k), src2(k2))
              k += 1
            }
          }
          ((i,j), new BDM(rows, cols, dst, 0, m1.majorStride, m1.isTranspose))
        }
      }
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map3(hbm2: M, hbm3: M, op: (Double, Double, Double) => Double): M = ???
  // {
  //   requireZippable(hbm2)
  //   requireZippable(hbm3)
  //   val blocks2 = blocks.join(hbm2.blocks).join(hbm3.blocks).mapValues { case ((m1, m2), m3) =>
  //     val src1 = m1.data
  //     val src2 = m2.data
  //     val src3 = m3.data
  //     val dst = new Array[Double](src1.length)
  //     var i = 0
  //     while (i < src1.length) {
  //       dst(i) = op(src1(i), src2(i), src3(i))
  //       i += 1
  //     }
  //     new BDM(m1.rows, m1.cols, dst)
  //   }
  //   new HailBlockMatrix(blocks2, blockSize, rows, cols)
  // }

  def map4(hbm2: M, hbm3: M, hbm4: M, op: (Double, Double, Double, Double) => Double): M = {
    requireZippable(hbm2)
    requireZippable(hbm3)
    requireZippable(hbm4)
    val blocks2 = blocks.zipPartitions(hbm2.blocks, hbm3.blocks, hbm4.blocks, preservesPartitioning = true) { (it1, it2, it3, it4) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = (it1.hasNext, it2.hasNext, it3.hasNext, it4.hasNext) match {
          case (true, true, true, true) => true
          case (false, false, false, false) => false
          case _ => throw new RuntimeException("Can only zip RDDs with " +
              "same number of elements in each partition")
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((i,j), m1) = it1.next()
          val ((i2,j2), m2) = it2.next()
          val ((i3,j3), m3) = it3.next()
          val ((i4,j4), m4) = it4.next()
          assert(m1.offset == 0, s"${m1.offset}")
          assert(m1.majorStride == (if (m1.isTranspose) m1.cols else m1.rows), s"${m1.majorStride} ${m1.isTranspose} ${m1.rows} ${m1.cols}")
          assert(m2.offset == 0, s"${m2.offset}")
          assert(m2.majorStride == (if (m2.isTranspose) m2.cols else m2.rows), s"${m2.majorStride} ${m2.isTranspose} ${m2.rows} ${m2.cols}")
          assert(m3.offset == 0, s"${m3.offset}")
          assert(m3.majorStride == (if (m3.isTranspose) m3.cols else m3.rows), s"${m3.majorStride} ${m3.isTranspose} ${m3.rows} ${m3.cols}")
          assert(m4.offset == 0, s"${m4.offset}")
          assert(m4.majorStride == (if (m4.isTranspose) m4.cols else m4.rows), s"${m4.majorStride} ${m4.isTranspose} ${m4.rows} ${m4.cols}")
          assert(i == i2, s"$i $i2")
          assert(j == j2, s"$j $j2")
          assert(i == i3, s"$i $i3")
          assert(j == j3, s"$j $j3")
          assert(i == i4, s"$i $i4")
          assert(j == j4, s"$j $j4")
          val rows = m1.rows
          val cols = m1.cols
          val src1 = m1.data
          val src2 = m2.data
          val src3 = m3.data
          val src4 = m4.data
          val dst = new Array[Double](src1.length)
          var k = 0
          if (m1.isTranspose == m2.isTranspose
            && m1.isTranspose == m3.isTranspose
            && m1.isTranspose == m4.isTranspose) {
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k), src3(k), src4(k))
              k += 1
            }
          } else {
            // FIXME: code gen the optimal tree?
            // FIXME: code gen the optimal tree on driver?
            while (k < src1.length) {
              val ii = k % m1.majorStride
              val jj = k / m1.majorStride
              val v2 = if (m1.isTranspose == m2.isTranspose) src2(k) else src2(jj + ii * m2.majorStride)
              val v3 = if (m1.isTranspose == m3.isTranspose) src3(k) else src3(jj + ii * m3.majorStride)
              val v4 = if (m1.isTranspose == m4.isTranspose) src4(k) else src4(jj + ii * m4.majorStride)
              dst(k) = op(src1(k), v2, v3, v4)
              k += 1
            }
          }
          ((i, j), new BDM(rows, cols, dst, 0, m1.majorStride, m1.isTranspose))
        }
      }
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def mapWithIndex(op: (Long, Long, Double) => Double): M = {
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
          result(i + j * m.rows) = op(iprefix + i, jprefix + j, m(i, j))
          i += 1
        }
        j += 1
      }
      new BDM(m.rows, m.cols, result, 0, m.rows, false)
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }

  def map2WithIndex(other: M, op: (Long, Long, Double, Double) => Double): M = {
    requireZippable(other)
    val blockSize = this.blockSize
    val blocks2 = blocks.zipPartitions(other.blocks, preservesPartitioning = true) { (thisIter, otherIter) =>
      new Iterator[((Int, Int), BDM[Double])] {
        def hasNext: Boolean = (thisIter.hasNext, otherIter.hasNext) match {
          case (true, true) => true
          case (false, false) => false
          case _ => throw new RuntimeException("Can only zip RDDs with " +
            "same number of elements in each partition")
        }
        def next(): ((Int, Int), BDM[Double]) = {
          val ((blocki,blockj), m1) = thisIter.next()
          val ((blocki2,blockj2), m2) = otherIter.next()
          assert(blocki == blocki2, s"$blocki $blocki2")
          assert(blockj == blockj2, s"$blockj $blockj2")

          val iprefix = blocki.toLong * blockSize
          val jprefix = blockj.toLong * blockSize
          val size = m1.cols * m1.rows
          val result = new Array[Double](size)
          var j = 0
          while (j < m1.cols) {
            var i = 0
            while (i < m1.rows) {
              result(i + j * m1.rows) = op(iprefix + i, jprefix + j, m1(i, j), m2(i, j))
              i += 1
            }
            j += 1
          }
          ((blocki, blockj), new BDM(m1.rows, m1.cols, result, 0, m1.rows, false))
        }
      }
    }
    new HailBlockMatrix(blocks2, blockSize, rows, cols)
  }


  def toIndexedRowMatrix(): IndexedRowMatrix = {
    require(cols < Integer.MAX_VALUE)
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
      val ioffset = i * blockSize
      val joffset = j * blockSize

      for (k <- 0 until m.rows)
      yield (k + ioffset, (joffset, m(k, ::).inner.toArray))
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

  private def block(hbm: HailBlockMatrix, bmPartitions: Array[Partition], p: HailGridPartitioner, context: TaskContext, i: Int, j: Int): BDM[Double] = {
    val it = hbm.blocks
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
    val rowsInThisBlock: Int = if (row + 1 == rowBlocks && rowsRemainder != 0) rowsRemainder else blockSize
    val colsInThisBlock: Int = if (col + 1 == colBlocks && colsRemainder != 0) colsRemainder else blockSize

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
