package is.hail.distributedmatrix

import java.io._

import breeze.linalg.{DenseMatrix => BDM, _}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import is.hail._
import is.hail.annotations._
import is.hail.table.Table
import is.hail.expr.EvalContext
import is.hail.expr.types._
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD}
import is.hail.utils._
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.commons.lang3.StringUtils
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._

object BlockMatrix {
  type M = BlockMatrix
  val defaultBlockSize: Int = 4096

  def from(sc: SparkContext, lm: BDM[Double]): M =
    from(sc, lm, defaultBlockSize)

  def from(sc: SparkContext, lm: BDM[Double], blockSize: Int): M = {
    assertCompatibleLocalMatrix(lm)
    val gp = GridPartitioner(blockSize, lm.rows, lm.cols)
    val localBlocksBc = Array.tabulate(gp.numPartitions) { pi =>
      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      sc.broadcast(lm(iOffset until iOffset + blockNRows, jOffset until jOffset + blockNCols).copy)
    }

    val blocks = new RDD[((Int, Int), BDM[Double])](sc, Nil) {
      override val partitioner = Some(gp)

      def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(i => IntPartition(i))

      def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
        val pi = split.index
        Iterator((gp.blockCoordinates(pi), localBlocksBc(split.index).value))
      }
    }

    new BlockMatrix(blocks, blockSize, lm.rows, lm.cols)
  }

  def from(irm: IndexedRowMatrix): M =
    from(irm, defaultBlockSize)

  def from(irm: IndexedRowMatrix, blockSize: Int): M =
    irm.toHailBlockMatrix(blockSize)

  // uniform or Gaussian
  def random(hc: HailContext, nRows: Int, nCols: Int, blockSize: Int = defaultBlockSize,
    seed: Int = 0, gaussian: Boolean = false): M = {

    val gp = GridPartitioner(blockSize, nRows, nCols)

    val blocks = new RDD[((Int, Int), BDM[Double])](hc.sc, Nil) {
      override val partitioner = Some(gp)

      def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
        new Partition {
          def index: Int = pi
        })

      def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
        val pi = split.index
        val (i, j) = gp.blockCoordinates(pi)
        val blockSeed = seed + pi

        val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(blockSeed)))
        val rand = if (gaussian) randBasis.gaussian else randBasis.uniform

        Iterator(((i, j), BDM.rand[Double](gp.blockRowNRows(i), gp.blockColNCols(j), rand)))
      }
    }

    new BlockMatrix(blocks, blockSize, nRows, nCols)
  }

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b, c, d, f)

  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r, f)

  val metadataRelativePath = "/metadata.json"

  /**
    * Reads a BlockMatrix matrix written by {@code write} at location {@code uri}.
    **/
  def read(hc: HailContext, uri: String): M = {
    val hadoop = hc.hadoopConf

    if (!hadoop.exists(uri + "/_SUCCESS"))
      fatal("Write failed: no success indicator found")

    val BlockMatrixMetadata(blockSize, nRows, nCols) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr =>
        implicit val formats = defaultJSONFormats
        jackson.Serialization.read[BlockMatrixMetadata](isr)
      }

    val gp = GridPartitioner(blockSize, nRows, nCols)

    def readBlock(i: Int, is: InputStream): Iterator[((Int, Int), BDM[Double])] = {
      val dis = new DataInputStream(is)
      val bdm = RichDenseMatrixDouble.read(dis)
      dis.close()

      Iterator.single(gp.blockCoordinates(i), bdm)
    }

    val nPartitions = gp.numPartitions
    val d = digitsNeeded(nPartitions)
    val partFiles = Array.tabulate[String](nPartitions) { i => partFile(d, i) }

    val blocks = hc.readPartitions(uri, partFiles, readBlock, Some(gp))

    new BlockMatrix(blocks, blockSize, nRows, nCols)
  }

  private[distributedmatrix] def assertCompatibleLocalMatrix(lm: BDM[Double]) {
    assert(lm.offset == 0, s"${ lm.offset }")
    assert(lm.majorStride == (if (lm.isTranspose) lm.cols else lm.rows), s"${ lm.majorStride } ${ lm.isTranspose } ${ lm.rows } ${ lm.cols }}")
  }

  private[distributedmatrix] def block(dm: BlockMatrix, partitions: Array[Partition], partitioner: GridPartitioner, context: TaskContext, i: Int, j: Int): BDM[Double] = {
    val it = dm.blocks
      .iterator(partitions(partitioner.coordinatesBlock(i, j)), context)
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
case class BlockMatrixMetadata(blockSize: Int, nRows: Long, nCols: Long)

class BlockMatrix(val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val nRows: Long,
  val nCols: Long) extends Serializable {

  import BlockMatrix._

  private[distributedmatrix] val st: String = Thread.currentThread().getStackTrace().mkString("\n")

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[GridPartitioner])

  val partitioner: GridPartitioner = blocks.partitioner.get.asInstanceOf[GridPartitioner]

  def transpose(): M =
    new BlockMatrix(new BlockMatrixTransposeRDD(this), blockSize, nCols, nRows)

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
    new BlockMatrix(new BlockMatrixMultiplyRDD(this, that), blockSize, nRows, that.nCols)

  def multiply(lm: BDM[Double]): M = {
    require(nCols == lm.rows,
      s"incompatible matrix dimensions: ${ nRows } x ${ nCols } and ${ lm.rows } x ${ lm.cols }")
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
    require(v.length == nRows, s"vector length, ${ v.length }, must equal number of matrix rows, ${ nRows }; v: ${ v: IndexedSeq[Double] }, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, _, x) => x + vBc.value(i.toInt))
  }

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): M = {
    require(v.length == nRows, s"vector length, ${ v.length }, must equal number of matrix rows, ${ nRows }; v: ${ v: IndexedSeq[Double] }, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((i, _, x) => x * vBc.value(i.toInt))
  }

  def vectorAddToEveryRow(v: Array[Double]): M = {
    require(v.length == nCols, s"vector length, ${ v.length }, must equal number of matrix columns, ${ nCols }; v: ${ v: IndexedSeq[Double] }, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((_, j, x) => x + vBc.value(j.toInt))
  }

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): M = {
    require(v.length == nCols, s"vector length, ${ v.length }, must equal number of matrix columns, ${ nCols }; v: ${ v: IndexedSeq[Double] }, m: $this")
    val vBc = blocks.sparkContext.broadcast(v)
    mapWithIndex((_, j, x) => x * vBc.value(j.toInt))
  }

  /**
    * Write {@code this} to a Hadoop sequence file at location {@code uri}.
    **/
  def write(uri: String, optKeep: Option[Array[Int]] = None, forceRowMajor: Boolean = false) {
    
    val hadoop = blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    def writeBlock(i: Int, it: Iterator[((Int, Int), BDM[Double])], os: OutputStream): Int = {
      assert(it.hasNext)
      var bdm = it.next()._2
      assert(!it.hasNext)

      val dos = new DataOutputStream(os)
      bdm.write(dos, forceRowMajor)
      dos.close()

      1
    }

    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, nCols),
        os)
    }

    optKeep match {
      case Some(keep) =>
        blocks.subsetPartitions(keep).writePartitions(uri, writeBlock, Some(keep, partitioner.numPartitions))
      case None =>
        blocks.writePartitions(uri, writeBlock)
    }
    
    hadoop.writeTextFile(uri + "/_SUCCESS")(out => ())
  }
  
  def writeBand(uri: String, lowerBandwidth: Long, upperBandwidth: Long, forceRowMajor: Boolean) {
    val keep = partitioner.bandedBlocks(lowerBandwidth, upperBandwidth)
    write(uri, Some(keep), forceRowMajor)
  }

  def cache(): this.type = {
    blocks.cache()
    this
  }

  def persist(storageLevel: StorageLevel): this.type = {
    blocks.persist(storageLevel)
    this
  }

  def persist(storageLevel: String): this.type = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }
    persist(level)
  }

  def unpersist(): this.type = {
    blocks.unpersist()
    this
  }

  def toLocalMatrix(): BDM[Double] = {
    require(this.nRows <= Int.MaxValue, "The number of rows of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently numRows: ${ this.nRows }")
    require(this.nCols <= Int.MaxValue, "The number of columns of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently numCols: ${ this.nCols }")
    require(this.nRows * this.nCols <= Int.MaxValue, "The length of the values array must be " +
      s"less than or equal to Int.MaxValue. Currently rows * cols: ${ this.nRows * this.nCols }")
    val nRows = this.nRows.toInt
    val nCols = this.nCols.toInt
    val localBlocks = blocks.collect()
    val values = new Array[Double](nRows * nCols)
    var bi = 0
    while (bi < localBlocks.length) {
      val ((i, j), lm) = localBlocks(bi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      lm.foreachPair { case ((ii, jj), v) =>
        values((jOffset + jj) * nRows + iOffset + ii) = v
      }
      bi += 1
    }
    new BDM(nRows, nCols, values)
  }

  private def requireZippable(that: M) {
    require(nRows == that.nRows,
      s"must have same number of rows, but actually: ${ nRows }x${ nCols }, ${ that.nRows }x${ that.nCols }")
    require(nCols == that.nCols,
      s"must have same number of cols, but actually: ${ nRows }x${ nCols }, ${ that.nRows }x${ that.nCols }")
    require(blockSize == that.blockSize,
      s"blocks must be same size, but actually were ${ blockSize }x${ blockSize } and ${ that.blockSize }x${ that.blockSize }")
  }

  def blockMap(op: BDM[Double] => BDM[Double]): M =
    new BlockMatrix(blocks.mapValues(op), blockSize, nRows, nCols)

  def blockMap2(that: M, op: (BDM[Double], BDM[Double]) => BDM[Double]): M = {
    requireZippable(that)
    new BlockMatrix(blocks.join(that.blocks).mapValues(op.tupled), blockSize, nRows, nCols)
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
    new BlockMatrix(blocks, blockSize, nRows, nCols)
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
          val ((i1, j1), lm1) = thisIter.next()
          val ((i2, j2), lm2) = thatIter.next()
          assertCompatibleLocalMatrix(lm1)
          assertCompatibleLocalMatrix(lm2)
          assert(i1 == i2, s"$i1 $i2")
          assert(j1 == j2, s"$j1 $j2")
          val nRows = lm1.rows
          val nCols = lm1.cols
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
          ((i1, j1), new BDM(nRows, nCols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new BlockMatrix(blocks, blockSize, nRows, nCols)
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
          val ((i1, j1), lm1) = it1.next()
          val ((i2, j2), lm2) = it2.next()
          val ((i3, j3), lm3) = it3.next()
          val ((i4, j4), lm4) = it4.next()
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
          val nRows = lm1.rows
          val nCols = lm1.cols
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
          ((i1, j1), new BDM(nRows, nCols, dst, 0, lm1.majorStride, lm1.isTranspose))
        }
      }
    }
    new BlockMatrix(blocks, blockSize, nRows, nCols)
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
    new BlockMatrix(blocks, blockSize, nRows, nCols)
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
          val ((i1, j1), lm1) = thisIter.next()
          val ((i2, j2), lm2) = thatIter.next()
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
    new BlockMatrix(blocks, blockSize, nRows, nCols)
  }

  def toIndexedRowMatrix(): IndexedRowMatrix = {
    require(this.nCols <= Integer.MAX_VALUE)
    val nCols = this.nCols.toInt

    def seqOp(a: Array[Double], p: (Int, Array[Double])): Array[Double] = p match {
      case (offset, v) =>
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
    }.aggregateByKey(new Array[Double](nCols))(seqOp, combOp)
      .map { case (i, a) => IndexedRow(i, new DenseVector(a)) },
      nRows, nCols)
  }

  def filterRows(keep: Array[Long]): BlockMatrix = this.transpose().filterCols(keep).transpose()

  def filterCols(keep: Array[Long]): BlockMatrix =
    new BlockMatrix(new BlockMatrixFilterColsRDD(this, keep), blockSize, nRows, keep.length)

  def filter(keepRows: Array[Long], keepCols: Array[Long]): BlockMatrix =
    new BlockMatrix(new BlockMatrixFilterRDD(this, keepRows, keepCols),
      blockSize, keepRows.length, keepCols.length)


  def entriesTable(hc: HailContext): Table = {
    val rvRowType = TStruct("i" -> TInt64Required, "j" -> TInt64Required, "entry" -> TFloat64Required)
    val entriesRDD = blocks.flatMap { case ((blockRow, blockCol), block) =>
      val rowOffset = blockRow * blockSize.toLong
      val colOffset = blockCol * blockSize.toLong

      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      block.activeIterator
        .map { case ((i, j), entry) =>
          region.clear()
          rvb.start(rvRowType)
          rvb.startStruct()
          rvb.addLong(rowOffset + i)
          rvb.addLong(colOffset + j)
          rvb.addDouble(entry)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
    }

    new Table(hc, entriesRDD, rvRowType)
  }
}

case class BlockMatrixFilterRDDPartition(index: Int,
  blockRowRanges: Array[(Int, Array[Int], Array[Int])],
  blockColRanges: Array[(Int, Array[Int], Array[Int])]) extends Partition

object BlockMatrixFilterRDD {
  // allBlockColRanges(newBlockCol) has elements of the form (blockCol, startIndices, endIndices) with blockCol increasing
  //   startIndices.zip(endIndices) gives all column-index ranges in blockCol to be copied to ranges in newBlockCol
  def computeAllBlockColRanges(keep: Array[Long],
    gp: GridPartitioner,
    newGP: GridPartitioner): Array[Array[(Int, Array[Int], Array[Int])]] = {

    val blockSize = gp.blockSize
    val ab = new ArrayBuilder[(Int, Array[Int], Array[Int])]()
    val startIndices = new ArrayBuilder[Int]()
    val endIndices = new ArrayBuilder[Int]()

    keep
      .grouped(blockSize)
      .zipWithIndex
      .map { case (colsInNewBlock, newBlockCol) =>
        ab.clear()

        val newBlockNCols = newGP.blockColNCols(newBlockCol)

        var j = 0 // start index in newBlockCol
      var k = 0 // end index in newBlockCol
        while (j < newBlockNCols) {
          startIndices.clear()
          endIndices.clear()

          var startCol = colsInNewBlock(j)
          val blockCol = (startCol / blockSize).toInt
          val finalColInBlockCol = blockCol * blockSize + gp.blockColNCols(blockCol)

          while (j < newBlockNCols && colsInNewBlock(j) < finalColInBlockCol) { // compute ranges for this blockCol
            val startCol = colsInNewBlock(j)
            val startColIndex = (startCol % blockSize).toInt // start index in blockCol
            startIndices += startColIndex

            var endCol = startCol + 1
            var k = j + 1
            while (k < newBlockNCols && colsInNewBlock(k) == endCol && endCol < finalColInBlockCol) { // extend range
              endCol += 1
              k += 1
            }
            endIndices += ((endCol - 1) % blockSize + 1).toInt // end index in blockCol
            j = k
          }
          ab += (blockCol, startIndices.result(), endIndices.result())
        }
        ab.result()
      }.toArray
  }

  def computeAllBlockRowRanges(keep: Array[Long],
    gp: GridPartitioner,
    newGP: GridPartitioner): Array[Array[(Int, Array[Int], Array[Int])]] = {

    computeAllBlockColRanges(keep, gp.transpose, newGP.transpose)
  }
}

private class BlockMatrixFilterRDD(dm: BlockMatrix, keepRows: Array[Long], keepCols: Array[Long])
  extends RDD[((Int, Int), BDM[Double])](dm.blocks.sparkContext, Nil) {
  require(keepRows.nonEmpty && keepRows.isIncreasing && keepRows.head >= 0 && keepRows.last < dm.nRows)
  require(keepCols.nonEmpty && keepCols.isIncreasing && keepCols.head >= 0 && keepCols.last < dm.nCols)

  private val gp = dm.partitioner
  private val blockSize = gp.blockSize
  private val newGP = GridPartitioner(blockSize, keepRows.length, keepCols.length)

  private val allBlockRowRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockRowRanges(keepRows, gp, newGP)

  private val allBlockColRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockColRanges(keepCols, gp, newGP)

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { pi =>
      BlockMatrixFilterRDDPartition(pi,
        allBlockRowRanges(newGP.blockBlockRow(pi)),
        allBlockColRanges(newGP.blockBlockCol(pi)))
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(dm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val (newBlockRow, newBlockCol) = newGP.blockCoordinates(partitionId)

        for {
          blockRow <- allBlockRowRanges(newBlockRow).map(_._1)
          blockCol <- allBlockColRanges(newBlockCol).map(_._1)
        } yield gp.coordinatesBlock(blockRow, blockCol)
      }
    })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val part = split.asInstanceOf[BlockMatrixFilterRDDPartition]

    val (newBlockRow, newBlockCol) = newGP.blockCoordinates(split.index)
    val (newBlockNRows, newBlockNCols) = newGP.blockDims(split.index)
    val newBlock = BDM.zeros[Double](newBlockNRows, newBlockNCols)

    var jCol = 0
    var kCol = 0
    part.blockColRanges.foreach { case (blockCol, colStartIndices, colEndIndices) =>
      val jCol0 = jCol // record first col index in newBlock corresponding to new blockCol
    var jRow = 0
      var kRow = 0
      part.blockRowRanges.foreach { case (blockRow, rowStartIndices, rowEndIndices) =>
        val jRow0 = jRow // record first row index in newBlock corresponding to new blockRow

        val parentPI = gp.coordinatesBlock(blockRow, blockCol)
        val (_, block) = dm.blocks.iterator(dm.blocks.partitions(parentPI), context).next()

        jCol = jCol0 // reset col index for new blockRow in same blockCol        
      var colRangeIndex = 0
        while (colRangeIndex < colStartIndices.length) {
          val siCol = colStartIndices(colRangeIndex)
          val eiCol = colEndIndices(colRangeIndex)
          kCol = jCol + eiCol - siCol

          jRow = jRow0 // reset row index for new column range in same (blockRow, blockCol)
          var rowRangeIndex = 0
          while (rowRangeIndex < rowStartIndices.length) {
            val siRow = rowStartIndices(rowRangeIndex)
            val eiRow = rowEndIndices(rowRangeIndex)
            kRow = jRow + eiRow - siRow

            newBlock(jRow until kRow, jCol until kCol) := block(siRow until eiRow, siCol until eiCol)

            jRow = kRow
            rowRangeIndex += 1
          }
          jCol = kCol
          colRangeIndex += 1
        }
      }
      assert(jRow == newBlockNRows)
    }
    assert(jCol == newBlockNCols)

    Iterator.single(((newBlockRow, newBlockCol), newBlock))
  }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

case class BlockMatrixFilterColsRDDPartition(index: Int, blockColRanges: Array[(Int, Array[Int], Array[Int])]) extends Partition

private class BlockMatrixFilterColsRDD(dm: BlockMatrix, keep: Array[Long])
  extends RDD[((Int, Int), BDM[Double])](dm.blocks.sparkContext, Nil) {
  require(keep.nonEmpty && keep.isIncreasing && keep.head >= 0 && keep.last < dm.nCols)

  private val gp = dm.partitioner
  private val blockSize = gp.blockSize
  private val newGP = GridPartitioner(blockSize, gp.nRows, keep.length)

  private val allBlockColRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockColRanges(keep, gp, newGP)

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { pi =>
      BlockMatrixFilterColsRDDPartition(pi, allBlockColRanges(newGP.blockBlockCol(pi)))
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(dm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val (blockRow, newBlockCol) = newGP.blockCoordinates(partitionId)
        allBlockColRanges(newBlockCol).map { case (blockCol, _, _) =>
          gp.coordinatesBlock(blockRow, blockCol)
        }
      }
    })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (blockRow, newBlockCol) = newGP.blockCoordinates(split.index)
    val (blockNRows, newBlockNCols) = newGP.blockDims(split.index)
    val newBlock = BDM.zeros[Double](blockNRows, newBlockNCols)

    var j = 0
    var k = 0
    split.asInstanceOf[BlockMatrixFilterColsRDDPartition]
      .blockColRanges
      .foreach { case (blockCol, startIndices, endIndices) =>
        val parentPI = gp.coordinatesBlock(blockRow, blockCol)
        val (_, block) = dm.blocks.iterator(dm.blocks.partitions(parentPI), context).next()

        var colRangeIndex = 0
        while (colRangeIndex < startIndices.length) {
          val si = startIndices(colRangeIndex)
          val ei = endIndices(colRangeIndex)
          k = j + ei - si

          newBlock(::, j until k) := block(::, si until ei)

          j = k
          colRangeIndex += 1
        }
      }
    assert(j == newBlockNCols)

    Iterator.single(((blockRow, newBlockCol), newBlock))
  }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

case class BlockMatrixTransposeRDDPartition(index: Int, prevPartition: Partition) extends Partition

private class BlockMatrixTransposeRDD(dm: BlockMatrix)
  extends RDD[((Int, Int), BDM[Double])](dm.blocks.sparkContext, Nil) {

  private val newPartitioner = dm.partitioner.transpose

  def transposePI(pi: Int): Int = dm.partitioner.coordinatesBlock(
    newPartitioner.blockBlockCol(pi),
    newPartitioner.blockBlockRow(pi))

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(dm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = Array(transposePI(partitionId))
    })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
    dm.blocks.iterator(split.asInstanceOf[BlockMatrixTransposeRDDPartition].prevPartition, context)
      .map { case ((j, i), lm) => ((i, j), lm.t) }

  protected def getPartitions: Array[Partition] = {
    Array.tabulate(newPartitioner.numPartitions) { pi =>
      BlockMatrixTransposeRDDPartition(pi, dm.blocks.partitions(transposePI(pi)))
    }
  }

  @transient override val partitioner: Option[Partitioner] = Some(newPartitioner)
}

private class BlockMatrixDiagonalRDD(m: BlockMatrix)
  extends RDD[Array[Double]](m.blocks.sparkContext, Nil) {

  import BlockMatrix.block

  private val length = {
    val x = math.min(m.nRows, m.nCols)
    assert(x <= Integer.MAX_VALUE, s"diagonal is too big for local array: $x; ${ m.st }")
    x.toInt
  }
  private val blockSize = m.blockSize
  private val dmPartitions = m.blocks.partitions
  private val dmPartitioner = m.partitioner
  private val partitionsLength = math.min(
    dmPartitioner.nBlockRows, dmPartitioner.nBlockCols)

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(m.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        assert(partitionId == 0)
        val deps = new Array[Int](partitionsLength)
        var i = 0
        while (i < partitionsLength) {
          deps(i) = dmPartitioner.coordinatesBlock(i, i)
          i += 1
        }
        deps
      }
    })

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

  require(l.nCols == r.nRows,
    s"inner dimensions must match, but given: ${ l.nRows }x${ l.nCols }, ${ r.nRows }x${ r.nCols }")
  require(l.blockSize == r.blockSize,
    s"blocks must be same size, but actually were ${ l.blockSize }x${ l.blockSize } and ${ r.blockSize }x${ r.blockSize }")

  private val lPartitioner = l.partitioner
  private val lPartitions = l.blocks.partitions
  private val rPartitioner = r.partitioner
  private val rPartitions = r.blocks.partitions
  private val nProducts = lPartitioner.nBlockCols
  private val gp = GridPartitioner(l.blockSize, l.nRows, r.nCols)

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val i = gp.blockBlockRow(partitionId)
          val deps = new Array[Int](nProducts)
          var k = 0
          while (k < nProducts) {
            deps(k) = lPartitioner.coordinatesBlock(i, k)
            k += 1
          }
          deps
        }
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val j = gp.blockBlockCol(partitionId)
          val deps = new Array[Int](nProducts)
          var k = 0
          while (k < nProducts) {
            deps(k) = rPartitioner.coordinatesBlock(k, j)
            k += 1
          }
          deps
        }
      })

  private def leftBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(l, lPartitions, lPartitioner, context, i, j)

  private def rightBlock(i: Int, j: Int, context: TaskContext): BDM[Double] =
    block(r, rPartitions, rPartitioner, context, i, j)

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (i, j) = gp.blockCoordinates(split.index)
    val (blockNRows, blockNCols) = gp.blockDims(split.index)
    val product = BDM.zeros[Double](blockNRows, blockNCols)
    var k = 0
    while (k < nProducts) {
      product :+= leftBlock(i, k, context) * rightBlock(k, j, context)
      k += 1
    }

    Iterator.single(((i, j), product))
  }

  protected def getPartitions: Array[Partition] =
    (0 until gp.numPartitions).map(IntPartition).toArray[Partition]

  @transient override val partitioner: Option[Partitioner] =
    Some(gp)
}

case class IntPartition(index: Int) extends Partition


// On compute, WriteBlocksRDDPartition writes the block row with index `index`
// [`start`, `end`] is the range of indices of parent partitions overlapping this block row
// `skip` is the index in the start partition corresponding to the first row of this block row
case class WriteBlocksRDDPartition(index: Int, start: Int, skip: Int, end: Int) extends Partition {
  def range: Range = start to end
}

class WriteBlocksRDD(path: String,
  rvd: RVD,
  sc: SparkContext,
  matrixType: MatrixType,
  sampleAnnotationsBc: Broadcast[IndexedSeq[Annotation]],
  parentPartStarts: Array[Long],
  f: () => java.lang.Double,
  ec: EvalContext,
  gp: GridPartitioner) extends RDD[Int](sc, Nil) {

  require(gp.nRows == parentPartStarts.last)

  private val parentParts = rvd.partitions
  private val blockSize = gp.blockSize

  private val d = digitsNeeded(gp.numPartitions)
  private val sHadoopBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(rvd.rdd) {
        def getParents(partitionId: Int): Seq[Int] =
          partitions(partitionId).asInstanceOf[WriteBlocksRDDPartition].range
      }
    )

  protected def getPartitions: Array[Partition] = {
    val nRows = parentPartStarts.last
    assert(nRows == gp.nRows)
    val nBlockRows = gp.nBlockRows

    val parts = new Array[Partition](nBlockRows)

    var firstRowInBlock = 0L
    var firstRowInNextBlock = 0L
    var pi = 0 // parent partition index
    var blockRow = 0
    while (blockRow < nBlockRows) {
      val skip = (firstRowInBlock - parentPartStarts(pi)).toInt

      firstRowInNextBlock = if (blockRow < nBlockRows - 1) firstRowInBlock + gp.blockSize else nRows

      val start = pi
      while (parentPartStarts(pi) < firstRowInNextBlock)
        pi += 1
      val end = pi - 1

      // if last parent partition overlaps next blockRow, don't advance
      if (parentPartStarts(pi) > firstRowInNextBlock)
        pi -= 1

      parts(blockRow) = WriteBlocksRDDPartition(blockRow, start, skip, end)

      firstRowInBlock = firstRowInNextBlock
      blockRow += 1
    }

    parts
  }

  def compute(split: Partition, context: TaskContext): Iterator[Int] = {
    val blockRow = split.index
    val nRowsInBlock = gp.blockRowNRows(blockRow)

    val dosPerBlockCol = Array.tabulate(gp.nBlockCols) { blockCol =>
      val nColsInBlock = gp.blockColNCols(blockCol)

      val i = gp.coordinatesBlock(blockRow, blockCol)
      val filename = path + "/parts/" + partFile(d, i)

      val dos = new DataOutputStream(sHadoopBc.value.value.unsafeWriter(filename))
      dos.writeInt(nRowsInBlock)
      dos.writeInt(nColsInBlock)
      dos.writeBoolean(true) // transposed, stored row major

      dos
    }

    val bytes = new Array[Byte](blockSize << 3)

    val entriesIndex = matrixType.entriesIdx
    val fullRow = new UnsafeRow(matrixType.rvRowType)
    val row = fullRow.deleteField(entriesIndex)

    val writeBlocksPart = split.asInstanceOf[WriteBlocksRDDPartition]
    val start = writeBlocksPart.start
    writeBlocksPart.range.foreach { pi =>
      val it = rvd.rdd.iterator(parentParts(pi), context)

      if (pi == start) {
        var j = 0
        while (j < writeBlocksPart.skip) {
          it.next()
          j += 1
        }
      }

      var i = 0
      while (it.hasNext && i < nRowsInBlock) {
        val rv = it.next()
        fullRow.set(rv)
        ec.set(1, row)
        val gs = fullRow.getAs[IndexedSeq[Any]](entriesIndex)
        var blockCol = 0
        var sampleIndex = 0
        while (blockCol < gp.nBlockCols) {
          val n = gp.blockColNCols(blockCol)
          var j = 0
          while (j < n) {
            ec.set(2, sampleAnnotationsBc.value(sampleIndex))
            ec.set(3, gs(sampleIndex))
            f() match {
              case null => fatal(s"Entry expr must be non-missing. Found missing value for col $j and row $row}")
              case t => Memory.storeDouble(bytes, j << 3, t.toDouble)
            }
            sampleIndex += 1
            j += 1
          }
          dosPerBlockCol(blockCol).write(bytes, 0, n << 3)

          blockCol += 1
        }
        i += 1
      }
    }

    dosPerBlockCol.foreach(_.close())

    Iterator.single(gp.nBlockCols) // number of blocks written
  }
}
