package is.hail.distributedmatrix

import java.io._

import breeze.linalg.{DenseMatrix => BDM, _}
import is.hail._
import is.hail.utils._
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.spark._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s._

object BlockMatrix {
  type M = BlockMatrix
  val defaultBlockSize: Int = 1024

  def from(sc: SparkContext, lm: BDM[Double]): M =
    from(sc, lm, defaultBlockSize)

  def from(sc: SparkContext, lm: BDM[Double], blockSize: Int): M = {
    assertCompatibleLocalMatrix(lm)
    val partitioner = GridPartitioner(blockSize, lm.rows, lm.cols)
    val lmBc = sc.broadcast(lm)
    val rowPartitions = partitioner.rowPartitions
    val colPartitions = partitioner.colPartitions
    val truncatedBlockRow = lm.rows / blockSize
    val truncatedBlockCol = lm.cols / blockSize
    val excessRows = lm.rows % blockSize
    val excessCols = lm.cols % blockSize
    val indices = for {
      i <- 0 until rowPartitions
      j <- 0 until colPartitions
    } yield (i, j)
    val blocks = sc.parallelize(indices).map { case (i, j) =>
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      val rowsInBlock = if (i == truncatedBlockRow) excessRows else blockSize
      val colsInBlock = if (j == truncatedBlockCol) excessCols else blockSize
      val a = new Array[Double](rowsInBlock * colsInBlock)
      var jj = 0
      while (jj < colsInBlock) {
        var ii = 0
        while (ii < rowsInBlock) {
          a(jj * rowsInBlock + ii) = lmBc.value(iOffset + ii, jOffset + jj)
          ii += 1
        }
        jj += 1
      }
      ((i, j), new BDM(rowsInBlock, colsInBlock, a))
    }.partitionBy(partitioner)
    new BlockMatrix(blocks, blockSize, lm.rows, lm.cols)
  }

  def from(irm: IndexedRowMatrix): M =
    from(irm, defaultBlockSize)

  def from(irm: IndexedRowMatrix, blockSize: Int): M =
    irm.toHailBlockMatrix(blockSize)

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b, c, d, f)

  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r, f)  
  
  private val metadataRelativePath = "/metadata.json"
  
  /**
    * Reads a BlockMatrix matrix written by {@code write} at location {@code uri}.
    **/
  def read(hc: HailContext, uri: String): M = {
    val hadoop = hc.hadoopConf
    hadoop.mkDir(uri)
    
    val BlockMatrixMetadata(blockSize, rows, cols, transposed) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr  =>
        jackson.Serialization.read[BlockMatrixMetadata](isr)
      }
    
    val gp = GridPartitioner(blockSize, rows, cols, transposed)    
    
    def readBlock(i: Int, is: InputStream): Iterator[((Int, Int), BDM[Double])] = {
      val dis = new DataInputStream(is)
      val bdm = RichDenseMatrixDouble.read(dis)
      dis.close()

      Iterator.single(gp.blockCoordinates(i), bdm)
    }
    
    val blocks = hc.readPartitions(uri, gp.numPartitions, readBlock, Some(gp))
    
    new BlockMatrix(blocks, blockSize, rows, cols)
  } 

  private[distributedmatrix] def assertCompatibleLocalMatrix(lm: BDM[Double]) {
    assert(lm.offset == 0, s"${lm.offset}")
    assert(lm.majorStride == (if (lm.isTranspose) lm.cols else lm.rows), s"${lm.majorStride} ${lm.isTranspose} ${lm.rows} ${lm.cols}}")
  }

  private[distributedmatrix] def block(dm: BlockMatrix, partitions: Array[Partition], partitioner: GridPartitioner, context: TaskContext, i: Int, j: Int): BDM[Double] = {
    val it = dm.blocks
      .iterator(partitions(partitioner.partitionIdFromBlockIndices(i, j)), context)
    assert(it.hasNext)
    val v = it.next()._2
    assert(!it.hasNext)
    v
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

      def --+(v: Array[Double]): M =
        l.vectorAddToEveryRow(v)
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
        r.blockMap(ll / _)
      }
    }
  }
}

// must be top-level for Jackson to serialize correctly
case class BlockMatrixMetadata(blockSize: Int, rows: Long, cols: Long, transposed: Boolean)

class BlockMatrix(val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val rows: Long,
  val cols: Long) extends Serializable {
  import BlockMatrix._

  private[distributedmatrix] val st: String = Thread.currentThread().getStackTrace().mkString("\n")

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[GridPartitioner])

  val partitioner: GridPartitioner = blocks.partitioner.get.asInstanceOf[GridPartitioner]

  def transpose(): M =
    new BlockMatrix(new BlockMatrixTransposeRDD(this), blockSize, cols, rows)

  def diagonal(): Array[Double] =
    new BlockMatrixDiagonalRDD(this).toArray

  def add(that: M): M =
    blockMap2(that, _ + _)

  def subtract(that: M): M =
    blockMap2(that, _ - _)

  def pointwiseMultiply(that: M): M =
    blockMap2(that, _ :* _)

  def pointwiseDivide(that: M): M =
    blockMap2(that, _ :/ _)

  def multiply(that: M): M =
    new BlockMatrix(new BlockMatrixMultiplyRDD(this, that), blockSize, rows, that.cols)

  def multiply(lm: BDM[Double]): M = {
    require(cols == lm.rows,
      s"incompatible matrix dimensions: ${ rows } x ${ cols } and ${ lm.rows } x ${ lm.cols }")
    multiply(BlockMatrix.from(blocks.sparkContext, lm, blockSize))
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
    require(v.length == rows, s"vector length, ${ v.length }, must equal number of matrix rows, ${ rows }; v: ${v: IndexedSeq[Double]}, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, _, x) => x + vBc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): M = {
    require(v.length == rows, s"vector length, ${ v.length }, must equal number of matrix rows, ${ rows }; v: ${v: IndexedSeq[Double]}, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, _, x) => x * vBc.value(i.toInt))
  }

  def vectorAddToEveryRow(v: Array[Double]): M = {
    require(v.length == cols, s"vector length, ${ v.length }, must equal number of matrix columns, ${ cols }; v: ${v: IndexedSeq[Double]}, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((_, j, x) => x + vBc.value(j.toInt))
  }

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): M = {
    require(v.length == cols, s"vector length, ${ v.length }, must equal number of matrix columns, ${ cols }; v: ${v: IndexedSeq[Double]}, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((_, j, x) => x * vBc.value(j.toInt))
  }
  
  /**
  * Writes the matrix {@code m} to a Hadoop sequence file at location {@code uri}.
  **/
  def write(uri: String) {
    val hadoop = blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    def writeBlock(i: Int, it: Iterator[((Int, Int), BDM[Double])], os: OutputStream): Int = {
      assert(it.hasNext)
      val (_, bdm) = it.next()
      assert(!it.hasNext)
      
      val dos = new DataOutputStream(os)
      bdm.write(dos)
      dos.close()
      
      1
    }
    
    blocks.writePartitions(uri, writeBlock)
    
    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, rows, cols, partitioner.transposed),
        os)
    }
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
    require(this.rows <= Int.MaxValue, "The number of rows of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently numRows: ${ this.rows }")
    require(this.cols <= Int.MaxValue, "The number of columns of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently numCols: ${ this.cols }")
    require(this.rows * this.cols <= Int.MaxValue, "The length of the values array must be " +
      s"less than or equal to Int.MaxValue. Currently rows * cols: ${ this.rows * this.cols }")
    val rows = this.rows.toInt
    val cols = this.cols.toInt
    val localBlocks = blocks.collect()
    val values = new Array[Double](rows * cols)
    var bi = 0
    while (bi < localBlocks.length) {
      val ((i, j), lm) = localBlocks(bi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      lm.foreachPair { case ((ii, jj), v) =>
        values((jOffset + jj) * rows + iOffset + ii) = v
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

  def blockMap(op: BDM[Double] => BDM[Double]): M =
    new BlockMatrix(blocks.mapValues(op), blockSize, rows, cols)

  def blockMap2(that: M, op: (BDM[Double], BDM[Double]) => BDM[Double]): M = {
    requireZippable(that)
    new BlockMatrix(blocks.join(that.blocks).mapValues(op.tupled), blockSize, rows, cols)
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
    new BlockMatrix(blocks, blockSize, rows, cols)
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
          if (lm1.isTranspose == lm2.isTranspose) {
            var k = 0
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k))
              k += 1
            }
          } else {
            val length = src1.length
            var k1 = 0
            var k2 = 0
            while (k1 < length) {
              while (k2 < length) {
                dst(k1) = op(src1(k1), src2(k2))
                k1 += 1
                k2 += lm2.majorStride
              }
              k2 += 1 - length
            }
          }
          ((i1,j1), new BDM(rows, cols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new BlockMatrix(blocks, blockSize, rows, cols)
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
          if (lm1.isTranspose == lm2.isTranspose
            && lm1.isTranspose == lm3.isTranspose
            && lm1.isTranspose == lm4.isTranspose) {
            var k = 0
            while (k < src1.length) {
              dst(k) = op(src1(k), src2(k), src3(k), src4(k))
              k += 1
            }
          } else {
            // FIXME: code gen the optimal tree on driver?
            val length = src1.length
            val lm1MinorSize = length / lm1.majorStride
            var k1 = 0
            var kt = 0
            while (k1 < length) {
              while (kt < length) {
                val v2 = if (lm1.isTranspose == lm2.isTranspose) src2(k1) else src2(kt)
                val v3 = if (lm1.isTranspose == lm3.isTranspose) src3(k1) else src3(kt)
                val v4 = if (lm1.isTranspose == lm4.isTranspose) src4(k1) else src4(kt)
                dst(k1) = op(src1(k1), v2, v3, v4)
                k1 += 1
                kt += lm1MinorSize
              }
              kt += 1 - length
            }
          }
          ((i1, j1), new BDM(rows, cols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new BlockMatrix(blocks, blockSize, rows, cols)
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
      new BDM(lm.rows, lm.cols, result)
    }
    new BlockMatrix(blocks, blockSize, rows, cols)
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
          assert(i1 == i2, s"$i1 $i2")
          assert(j1 == j2, s"$j1 $j2")
          val iOffset = i1.toLong * blockSize
          val jOffset = j1.toLong * blockSize
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
          ((i1, j1), new BDM(lm1.rows, lm1.cols, result))
        }
      }
    }
    new BlockMatrix(blocks, blockSize, rows, cols)
  }

  def toIndexedRowMatrix(): IndexedRowMatrix = {
    require(this.cols <= Integer.MAX_VALUE)
    val cols = this.cols.toInt

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

    new IndexedRowMatrix(this.blocks.flatMap { case ((i, j), lm) =>
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      for (k <- 0 until lm.rows)
      yield (k + iOffset, (jOffset, lm(k, ::).inner.toArray))
    }.aggregateByKey(new Array[Double](cols))(seqOp, combOp)
      .map { case (i, a) => IndexedRow(i, new DenseVector(a)) },
      rows, cols)
  }
}

private class BlockMatrixTransposeRDD(dm: BlockMatrix)
  extends RDD[((Int, Int), BDM[Double])](dm.blocks.sparkContext, Seq[Dependency[_]](new OneToOneDependency(dm.blocks))) {
  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
    dm.blocks.iterator(split, context).map { case ((i, j), lm) => ((j, i), lm.t) }

  protected def getPartitions: Array[Partition] =
    dm.blocks.partitions

  private val prevPartitioner = dm.partitioner
  @transient override val partitioner: Option[Partitioner] =
    Some(prevPartitioner.transpose)
}

private class BlockMatrixDiagonalRDD(m: BlockMatrix)
    extends RDD[Array[Double]](m.blocks.sparkContext, Nil) {
  import BlockMatrix.block

  private val length = {
    val x = math.min(m.rows, m.cols)
    assert(x <= Integer.MAX_VALUE, s"diagonal is too big for local array: $x; ${m.st}")
    x.toInt
  }
  private val blockSize = m.blockSize
  private val dmPartitions = m.blocks.partitions
  private val dmPartitioner = m.partitioner
  private val partitionsLength = math.min(
    dmPartitioner.rowPartitions, dmPartitioner.colPartitions)

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(m.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        assert(partitionId == 0)
        val deps = new Array[Int](partitionsLength)
        var i = 0
        while (i < partitionsLength) {
          deps(i) = dmPartitioner.partitionIdFromBlockIndices(i, i)
          i += 1
        }
        deps
      }})

  def compute(split: Partition, context: TaskContext): Iterator[Array[Double]] = {
    val result = new Array[Double](length)
    var i = 0
    while (i < partitionsLength) {
      val a = diag(block(m, dmPartitions, dmPartitioner, context, i, i)).toArray
      var k = 0
      val offset = i * blockSize
      while (k < a.length) {
        result(offset + k) = a(k)
        k += 1
      }
      i += 1
    }
    Iterator.single(result)
  }

  protected def getPartitions: Array[Partition] =
    Array(IntPartition(0))

  def toArray: Array[Double] = {
    val a = this.collect()
    assert(a.length == 1)
    a(0)
  }
}

private class BlockMatrixMultiplyRDD(l: BlockMatrix, r: BlockMatrix)
    extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {
  import BlockMatrix.block

  require(l.cols == r.rows,
    s"inner dimensions must match, but given: ${ l.rows }x${ l.cols }, ${ r.rows }x${ r.cols }")
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
  private val truncatedBlockRow = rows / blockSize
  private val truncatedBlockCol = cols / blockSize
  private val nProducts = lPartitioner.colPartitions
  private val excessRows = (rows % blockSize).toInt
  private val excessCols = (cols % blockSize).toInt
  private val nPartitions = rowPartitions * colPartitions

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val row = _partitioner.blockRowIndex(partitionId)
          val deps = new Array[Int](nProducts)
          var j = 0
          while (j < nProducts) {
            deps(j) = lPartitioner.partitionIdFromBlockIndices(row, j)
            j += 1
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

  private def leftBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(l, lPartitions, lPartitioner, context, i, j)

  private def rightBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(r, rPartitions, rPartitioner, context, i, j)

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val i = _partitioner.blockRowIndex(split.index)
    val j = _partitioner.blockColIndex(split.index)
    val rowsInBlock: Int = if (i == truncatedBlockRow) excessRows else blockSize
    val colsInBlock: Int = if (j == truncatedBlockCol) excessCols else blockSize

    val product = BDM.zeros[Double](rowsInBlock, colsInBlock)
    var k = 0
    while (k < nProducts) {
      product :+= leftBlock(i, k, context) * rightBlock(k, j, context)
      k += 1
    }

    Iterator.single(((i, j), product))
  }

  protected def getPartitions: Array[Partition] =
    (0 until nPartitions).map(IntPartition).toArray[Partition]

  private val _partitioner = GridPartitioner(blockSize, rows, cols)
  /** Optionally overridden by subclasses to specify how they are partitioned. */
  @transient override val partitioner: Option[Partitioner] =
    Some(_partitioner)
}

case class IntPartition(index: Int) extends Partition

object WriteBlocksRDD {
  def allDependencies(boundaries: Array[Long], nBlockRows: Int, blockSize: Int): Array[Array[Int]] = {
    val deps = Array.ofDim[Array[Int]](nBlockRows)

    val ab = new ArrayBuilder[Int]()
    var i = 0 // original partition index
    var blockStart = 0L
    var blockEnd = 0L

    var block = 0
    while (block < nBlockRows) {
      blockStart = blockEnd
      if (block != nBlockRows - 1)
        blockEnd += blockSize
      else
        blockEnd = boundaries.last

      // build array of i such that [boundaries(i), boundaries(i + 1)) intersects [blockMin, blockEnd)      
      ab.clear()
      while (boundaries(i) < blockEnd && boundaries(i + 1) > blockStart) {
        ab += i
        i += 1
      }
      if (boundaries(i) > blockEnd) // if partition i extends into next block, don't advance to next partition
        i -= 1
      
      deps(block) = ab.result()
      block += 1
    }

    deps
  }
}

// FIXME: requires dense IRM, no missing rows
// FIXME: start with function on IRM, then make directly on VSM?
class WriteBlocksRDD(irm: IndexedRowMatrix, path: String, blockSize: Int) extends RDD[Int](irm.rows.sparkContext, Nil) {
  private val boundaries: Array[Long] = irm.rows.computeBoundaries()
  assert(irm.numRows() == boundaries.last) // can replace irm.numRows()

  private val rows = irm.numRows()
  private val cols = irm.numCols()
  private val gp = GridPartitioner(blockSize, rows, cols)
  private val rowPartitions: Int = gp.rowPartitions
  private val colPartitions: Int = gp.colPartitions
  private val truncatedBlockRow = rows / blockSize
  private val truncatedBlockCol = cols / blockSize
  private val excessRows = (rows % blockSize).toInt
  private val excessCols = (cols % blockSize).toInt
  
  private val allDeps = WriteBlocksRDD.allDependencies(boundaries, rowPartitions, blockSize)
  
  override def getDependencies: Seq[Dependency[_]] = {  
    Array[Dependency[_]](
      new NarrowDependency(irm.rows) {
        def getParents(partitionId: Int): Seq[Int] = allDeps(partitionId)
      }
    )
  }
  
  protected def getPartitions: Array[Partition] =
    (0 until rowPartitions).map(IntPartition).toArray[Partition]
  
  
  // FIXME: do we need to compute colPartitions times?
  def compute(split: Partition, context: TaskContext): Iterator[Int] = {
    val blockRowIndex = split.index
    
    val nRows =
      if (blockRowIndex == truncatedBlockRow)
        excessRows
      else
        blockSize
    
    val data = Array.ofDim[Double](nRows * blockSize)
    
    var blockColIndex = 0
    var j = 0
    while (blockColIndex < colPartitions) {
      val nCols = 
        if (blockRowIndex == truncatedBlockRow)
          excessCols
        else
          blockSize
      
      val lastCol = j + nCols
      
      while (j < lastCol) {
        
        //add to data
        
        j += 1
      }
      
      if (blockColIndex == truncatedBlockCol) {
        // copy out subarray
      } else
        //use whole array
      
      //form and write out BDM
      
      blockColIndex += 1
    }
    
    Iterator.single(blockColIndex)
  }
}