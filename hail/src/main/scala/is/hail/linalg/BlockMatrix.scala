package is.hail.linalg

import is.hail._
import is.hail.annotations._
import is.hail.backend.{BroadcastValue, ExecuteContext, HailStateManager}
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.expr.ir.{IntArrayBuilder, TableReader, TableValue, ThreefryRandomEngine}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.IndexWriter
import is.hail.rvd.{RVD, RVDContext}
import is.hail.sparkextras.{ContextRDD, OriginUnionPartition, OriginUnionRDD}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.richUtils.{
  ByteTrackingOutputStream, RichArray, RichContextRDD, RichDenseMatrixDouble,
}

import scala.collection.immutable.NumericRange

import java.io._

import breeze.linalg.{sum => breezeSum, DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics.{abs => breezeAbs, log => breezeLog, pow => breezePow, sqrt => breezeSqrt}
import breeze.stats.distributions.RandBasis
import org.apache.commons.lang3.StringUtils
import org.apache.spark._
import org.apache.spark.executor.InputMetrics
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s._

case class CollectMatricesRDDPartition(
  index: Int,
  firstPartition: Int,
  blockPartitions: Array[Partition],
  blockSize: Int,
  nRows: Int,
  nCols: Int,
) extends Partition {
  def nBlocks: Int = blockPartitions.length
}

class CollectMatricesRDD(@transient var bms: IndexedSeq[BlockMatrix])
    extends RDD[BDM[Double]](SparkBackend.sparkContext("CollectMatricesRDD"), Nil) {
  private val nBlocks = bms.map(_.blocks.getNumPartitions)
  private val firstPartition = nBlocks.scan(0)(_ + _).init

  protected def getPartitions: Array[Partition] =
    bms.iterator.zipWithIndex.map { case (bm, i) =>
      CollectMatricesRDDPartition(
        i,
        firstPartition(i),
        bm.blocks.partitions,
        bm.blockSize,
        bm.nRows.toInt,
        bm.nCols.toInt,
      )
    }
      .toArray

  override def getDependencies: Seq[Dependency[_]] =
    bms.zipWithIndex.map { case (bm, i) =>
      val n = nBlocks(i)
      new NarrowDependency(bm.blocks) {
        def getParents(j: Int): Seq[Int] =
          if (j == i)
            0 until n
          else
            FastSeq.empty
      }
    }

  def compute(split: Partition, context: TaskContext): Iterator[BDM[Double]] = {
    val p = split.asInstanceOf[CollectMatricesRDDPartition]
    val m = BDM.zeros[Double](p.nRows, p.nCols)
    val prev = parent[((Int, Int), BDM[Double])](p.index)
    var k = 0
    while (k < p.nBlocks) {
      val it = prev.iterator(p.blockPartitions(k), context)
      assert(it.hasNext)
      val ((i, j), b) = it.next()

      m(
        (i * p.blockSize) until (i * p.blockSize + b.rows),
        (j * p.blockSize) until (j * p.blockSize + b.cols),
      ) := b

      k += 1
    }

    Iterator.single(m)
  }

  override def clearDependencies(): Unit = {
    super.clearDependencies()
    bms = null
  }
}

object BlockMatrix {
  type M = BlockMatrix
  val defaultBlockSize: Int = 4096 // 32 * 1024 bytes
  val bufferSpecBlockSize = 32 * 1024

  val bufferSpec: BufferSpec =
    new BlockingBufferSpec(
      bufferSpecBlockSize,
      new LZ4FastBlockBufferSpec(bufferSpecBlockSize, new StreamBlockBufferSpec),
    )

  def apply(gp: GridPartitioner, piBlock: (GridPartitioner, Int) => ((Int, Int), BDM[Double]))
    : BlockMatrix =
    new BlockMatrix(
      new RDD[((Int, Int), BDM[Double])](SparkBackend.sparkContext("BlockMatrix.apply"), Nil) {
        override val partitioner = Some(gp)

        protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
          new Partition { def index: Int = pi }
        )

        def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
          Iterator.single(piBlock(gp, split.index))
      },
      gp.blockSize,
      gp.nRows,
      gp.nCols,
    )

  def fromBreezeMatrix(lm: BDM[Double]): M =
    fromBreezeMatrix(lm, defaultBlockSize)

  def fromBreezeMatrix(lm: BDM[Double], blockSize: Int): M = {
    val gp = GridPartitioner(blockSize, lm.rows, lm.cols)

    val localBlocksBc = Array.tabulate(gp.numPartitions) { pi =>
      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      HailContext.backend.broadcast(lm(
        iOffset until iOffset + blockNRows,
        jOffset until jOffset + blockNCols,
      ).copy)
    }

    BlockMatrix(gp, (gp, pi) => (gp.blockCoordinates(pi), localBlocksBc(pi).value))
  }

  def fromIRM(irm: IndexedRowMatrix): M =
    fromIRM(irm, defaultBlockSize)

  def fromIRM(irm: IndexedRowMatrix, blockSize: Int): M =
    irm.toHailBlockMatrix(blockSize)

  def fill(nRows: Long, nCols: Long, value: Double, blockSize: Int = defaultBlockSize)
    : BlockMatrix =
    BlockMatrix(
      GridPartitioner(blockSize, nRows, nCols),
      (gp, pi) => {
        val (i, j) = gp.blockCoordinates(pi)
        ((i, j), BDM.fill[Double](gp.blockRowNRows(i), gp.blockColNCols(j))(value))
      },
    )

  // uniform or Gaussian
  def random(
    nRows: Long,
    nCols: Long,
    blockSize: Int = defaultBlockSize,
    nonce: Long = 0,
    staticUID: Long = 0,
    gaussian: Boolean,
  ): M =
    BlockMatrix(
      GridPartitioner(blockSize, nRows, nCols),
      (gp, pi) => {
        val (i, j) = gp.blockCoordinates(pi)
        val generator = ThreefryRandomEngine(nonce, staticUID, Array(pi.toLong))
        val randBasis: RandBasis = new RandBasis(generator)
        val rand = if (gaussian) randBasis.gaussian else randBasis.uniform

        ((i, j), BDM.rand[Double](gp.blockRowNRows(i), gp.blockColNCols(j), rand))
      },
    )

  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r, f)

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b, c, d, f)

  val metadataRelativePath = "/metadata.json"

  def checkWriteSuccess(fs: FS, uri: String): Unit =
    if (!fs.isFile(uri + "/_SUCCESS"))
      fatal(
        s"Error reading block matrix. Earlier write failed: no success indicator found at uri $uri"
      )

  def readMetadata(fs: FS, uri: String): BlockMatrixMetadata =
    using(fs.open(uri + metadataRelativePath)) { is =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.read[BlockMatrixMetadata](is)
    }

  def read(fs: FS, uri: String): M = {
    checkWriteSuccess(fs, uri)

    val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) =
      readMetadata(fs, uri)

    val gp = GridPartitioner(blockSize, nRows, nCols, maybeFiltered)

    def readBlock(pi: Int, is: InputStream, metrics: InputMetrics)
      : Iterator[((Int, Int), BDM[Double])] = {
      val block = RichDenseMatrixDouble.read(is, bufferSpec)
      is.close()

      Iterator.single(gp.partCoordinates(pi) -> block)
    }

    val blocks = HailContext.readPartitions(fs, uri, partFiles, readBlock, Some(gp))

    new BlockMatrix(blocks, blockSize, nRows, nCols)
  }

  private[linalg] def assertCompatibleLocalMatrix(lm: BDM[Double]): Unit =
    assert(lm.isCompact)

  private[linalg] def block(
    bm: BlockMatrix,
    parts: Array[Partition],
    gp: GridPartitioner,
    context: TaskContext,
    i: Int,
    j: Int,
  ): Option[BDM[Double]] = {
    val pi = gp.coordinatesPart(i, j)
    if (pi >= 0) {
      val it = bm.blocks.iterator(parts(pi), context)
      assert(it.hasNext)
      val (_, lm) = it.next()
      assert(!it.hasNext)
      Some(lm)
    } else {
      None
    }
  }

  def negationOp: BDM[Double] => BDM[Double] = -_

  def reverseScalarDiv(r: BDM[Double], l: Double): BDM[Double] = l /:/ r

  object ops {

    implicit class Shim(l: M) {
      def +(r: M): M = l.add(r)
      def -(r: M): M = l.sub(r)
      def *(r: M): M = l.mul(r)
      def /(r: M): M = l.div(r)
      def +(r: Double): M = l.scalarAdd(r)
      def -(r: Double): M = l.scalarSub(r)
      def *(r: Double): M = l.scalarMul(r)
      def /(r: Double): M = l.scalarDiv(r)
      def T: M = l.transpose()
    }

    implicit class ScalarShim(l: Double) {
      def +(r: M): M = r.scalarAdd(l)
      def -(r: M): M = r.reverseScalarSub(l)
      def *(r: M): M = r.scalarMul(l)
      def /(r: M): M = r.reverseScalarDiv(l)
    }
  }

  def collectMatrices(bms: IndexedSeq[BlockMatrix]): RDD[BDM[Double]] = new CollectMatricesRDD(bms)

  def binaryWriteBlockMatrices(
    fs: FS,
    bms: IndexedSeq[BlockMatrix],
    prefix: String,
    overwrite: Boolean,
  ): Unit = {
    if (overwrite)
      fs.delete(prefix, recursive = true)
    else if (fs.exists(prefix))
      fatal(s"file already exists: $prefix")

    fs.mkDir(prefix)

    val d = digitsNeeded(bms.length)
    val fsBc = fs.broadcast
    val partitionCounts = collectMatrices(bms)
      .mapPartitionsWithIndex { case (i, it) =>
        assert(it.hasNext)
        val m = it.next()
        val path = prefix + "/" + StringUtils.leftPad(i.toString, d, '0')

        RichDenseMatrixDouble.exportToDoubles(fsBc.value, path, m, forceRowMajor = true)

        Iterator.single(1)
      }
      .collect()

    using(fs.create(prefix + "/_SUCCESS"))(out => ())
  }

  def exportBlockMatrices(
    fs: FS,
    bms: IndexedSeq[BlockMatrix],
    prefix: String,
    overwrite: Boolean,
    delimiter: String,
    header: Option[String],
    addIndex: Boolean,
    compression: Option[String],
    customFilenames: Option[Array[String]],
  ): Unit = {

    if (overwrite)
      fs.delete(prefix, recursive = true)
    else if (fs.exists(prefix))
      fatal(s"file already exists: $prefix")

    fs.mkDir(prefix)

    val d = digitsNeeded(bms.length)
    val fsBc = fs.broadcast

    val nameFunction = customFilenames match {
      case None => i: Int => StringUtils.leftPad(i.toString, d, '0') + ".tsv"
      case Some(filenames) => filenames.apply(_)
    }

    val compressionExtension = compression.map(x => "." + x).getOrElse("")

    val partitionCounts = collectMatrices(bms)
      .mapPartitionsWithIndex { case (i, it) =>
        assert(it.hasNext)
        val m = it.next()
        val path = prefix + "/" + nameFunction(i) + compressionExtension

        using(
          new PrintWriter(
            new BufferedWriter(
              new OutputStreamWriter(
                fsBc.value.create(path)
              )
            )
          )
        ) { f =>
          header.foreach(h => f.println(h))

          var i = 0
          while (i < m.rows) {
            if (addIndex) {
              f.print(i)
              f.print(delimiter)
            }

            var j = 0
            while (j < m.cols) {
              f.print(m(i, j))
              if (j < m.cols - 1)
                f.print(delimiter)
              j += 1
            }
            f.print('\n')
            i += 1
          }
        }

        Iterator.single(1)
      }
      .collect()

    using(fs.create(prefix + "/_SUCCESS"))(out => ())
  }

  def writeBlockMatrices(
    ctx: ExecuteContext,
    bms: IndexedSeq[BlockMatrix],
    prefix: String,
    overwrite: Boolean,
    forceRowMajor: Boolean,
  ): Unit = {

    def blockMatrixURI(matrixIdx: Int): String = prefix + "_" + matrixIdx
    val fs = ctx.fs
    val tmpdir = ctx.localTmpdir

    bms.zipWithIndex.foreach { case (_, bIdx) =>
      val uri = blockMatrixURI(bIdx)
      if (overwrite)
        fs.delete(uri, recursive = true)
      else if (fs.exists(uri))
        fatal(s"file already exists: $uri")

      fs.mkDir(uri)
      fs.mkDir(uri + "/parts")
    }

    def writeBlock(
      ctx: RVDContext,
      it: Iterator[((Int, Int), BDM[Double])],
      os: OutputStream,
      iw: IndexWriter,
    ): (Long, Long) = {
      val btos = new ByteTrackingOutputStream(os)
      assert(it.hasNext)
      val (_, lm) = it.next()
      assert(!it.hasNext)

      lm.write(btos, forceRowMajor, bufferSpec)
      val bytesWritten = btos.bytesWritten
      btos.close()

      (1L, bytesWritten)
    }

    if (bms.isEmpty) {
      return
    }

    val rdds = bms.map(bm => bm.blocks)
    val blockMatrixMetadataFields =
      bms.map(bm => (bm.blockSize, bm.nRows, bm.nCols, bm.gp.partitionIndexToBlockIndex))
    val first = rdds(0)
    val nPartitions = rdds.map(_.getNumPartitions).sum
    val numDigits = digitsNeeded(nPartitions)

    val ordd = new OriginUnionRDD[((Int, Int), BDM[Double]), ((Int, Int), BDM[Double])](
      first.sparkContext,
      rdds,
      (_, _, it) => it,
    )

    val partMap = ordd.partitions.map(part => part.asInstanceOf[OriginUnionPartition]).map(oup =>
      (oup.index, (oup.originIdx, oup.originPart.index))
    ).toMap

    val writerRDD = ContextRDD.weaken(ordd).cmapPartitionsWithContextAndIndex { (i, ctx, it) =>
      val (rddIndex, partIndex) = partMap(i)
      val trueIt = it(ctx)
      val rootPath = blockMatrixURI(rddIndex)
      val fileName = partFile(numDigits, partIndex, TaskContext.get)
      val fileDataIterator = RichContextRDD.writeParts(
        ctx,
        rootPath,
        fileName,
        null,
        (_, _) => null,
        false,
        fs,
        tmpdir,
        trueIt,
        writeBlock,
      )
      fileDataIterator.map(fd => (fd, rddIndex))
    }

    val rddNumberAndPartFiles = writerRDD.collect()
    val grouped = rddNumberAndPartFiles.groupBy(_._2)
    grouped.foreach { case (rddIndex, numberedPartFiles) =>
      val fileData = numberedPartFiles.map { case (partFileName, _) => partFileName }
      val metadataPath = blockMatrixURI(rddIndex.toInt) + metadataRelativePath
      using(new DataOutputStream(fs.create(metadataPath))) { os =>
        implicit val formats = defaultJSONFormats
        val (blockSize, nRows, nCols, maybeBlocks) = blockMatrixMetadataFields(rddIndex.toInt)
        jackson.Serialization.write(
          BlockMatrixMetadata(blockSize, nRows, nCols, maybeBlocks, fileData.map(_.path)),
          os,
        )
      }

      using(fs.create(blockMatrixURI(rddIndex.toInt) + "/_SUCCESS"))(out => ())
    }
  }
}

// must be top-level for Jackson to serialize correctly
case class BlockMatrixMetadata(
  blockSize: Int,
  nRows: Long,
  nCols: Long,
  maybeFiltered: Option[IndexedSeq[Int]],
  partFiles: IndexedSeq[String],
)

class BlockMatrix(
  val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val nRows: Long,
  val nCols: Long,
) extends Serializable {

  import BlockMatrix._

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[GridPartitioner])

  val gp: GridPartitioner = blocks.partitioner.get.asInstanceOf[GridPartitioner]

  require(gp.blockSize == blockSize && gp.nRows == nRows && gp.nCols == nCols)

  val isSparse: Boolean = gp.partitionIndexToBlockIndex.isDefined

  def requireDense(name: String): Unit =
    if (isSparse)
      fatal(s"$name is not supported for block-sparse matrices.")

  def densify(): BlockMatrix =
    if (isSparse) {
      require(gp.maxNBlocks <= Int.MaxValue)
      realizeBlocks(None)
    } else
      this

  // if Some(bis), unrealized blocks in bis are replaced with zero blocks
  // if None, all unrealized blocks are replaced with zero blocks
  def realizeBlocks(maybeBlocksToRealize: Option[IndexedSeq[Int]]): BlockMatrix = {
    val realizeGP = gp.copy(partitionIndexToBlockIndex =
      if (maybeBlocksToRealize.exists(_.length == gp.maxNBlocks)) None else maybeBlocksToRealize
    )

    val newGP = gp.union(realizeGP)

    if (newGP.numPartitions == gp.numPartitions)
      this
    else {
      def newPIPartition(pi: Int): Iterator[((Int, Int), BDM[Double])] = {
        val bi = newGP.partitionToBlock(pi)
        val lm = (BDM.zeros[Double] _).tupled(newGP.blockDims(bi))
        Iterator.single((newGP.blockCoordinates(bi), lm))
      }
      val oldToNewPI = gp.partitionIndexToBlockIndex.get.map(newGP.blockToPartition)
      val newBlocks =
        blocks.supersetPartitions(oldToNewPI, newGP.numPartitions, newPIPartition, Some(newGP))

      new BlockMatrix(newBlocks, blockSize, nRows, nCols)
    }
  }

  def filterBlocks(blocksToKeep: Array[Int]): BlockMatrix =
    if (blocksToKeep.length == gp.maxNBlocks)
      this
    else
      subsetBlocks(gp.intersect(gp.copy(partitionIndexToBlockIndex = Some(blocksToKeep))))

  // assumes subsetGP blocks are subset of gp blocks, as with subsetGP = gp.intersect(gp2)
  def subsetBlocks(subsetGP: GridPartitioner): BlockMatrix = {
    if (subsetGP.numPartitions == gp.numPartitions)
      this
    else {
      assert(subsetGP.partitionIndexToBlockIndex.isDefined)
      new BlockMatrix(
        blocks.subsetPartitions(
          subsetGP.partitionIndexToBlockIndex.get.map(gp.blockToPartition),
          Some(subsetGP),
        ),
        blockSize,
        nRows,
        nCols,
      )
    }
  }

  // filter to blocks overlapping diagonal band of all elements with lower <= jj - ii <= upper
  // if not blocksOnly, also zero out all remaining elements outside band
  def filterBand(lower: Long, upper: Long, blocksOnly: Boolean): BlockMatrix = {
    require(lower <= upper)

    val filteredBM = filterBlocks(gp.bandBlocks(lower, upper))

    if (blocksOnly)
      filteredBM
    else
      filteredBM.zeroBand(lower, upper)
  }

  def zeroBand(lower: Long, upper: Long): BlockMatrix = {
    val zeroedBlocks = blocks.mapPartitions(
      { it =>
        assert(it.hasNext)
        val ((i, j), lm0) = it.next()
        assert(!it.hasNext)

        val nRowsInBlock = lm0.rows
        val nColsInBlock = lm0.cols

        val diagIndex = (j - i).toLong * blockSize
        val lowestDiagIndex = diagIndex - (nRowsInBlock - 1)
        val highestDiagIndex = diagIndex + (nColsInBlock - 1)

        if (lowestDiagIndex >= lower && highestDiagIndex <= upper)
          Iterator.single(((i, j), lm0))
        else {
          val lm = lm0.copy // avoidable?

          if (lower > lowestDiagIndex) {
            val iiLeft = math.max(diagIndex - lower, 0).toInt
            val iiRight = math.min(diagIndex - lower + nColsInBlock, nRowsInBlock).toInt

            var ii = iiLeft
            var jj = math.max(lower - diagIndex, 0).toInt
            while (ii < iiRight) {
              lm(ii to ii, 0 until jj) := 0.0
              ii += 1
              jj += 1
            }

            lm(iiRight until nRowsInBlock, ::) := 0.0
          }

          if (upper < highestDiagIndex) {
            val iiLeft = math.max(diagIndex - upper, 0).toInt
            val iiRight = math.min(diagIndex - upper + nColsInBlock, nRowsInBlock).toInt

            lm(0 until iiLeft, ::) := 0.0

            var ii = iiLeft
            var jj = math.max(upper - diagIndex, 0).toInt + 1
            while (ii < iiRight) {
              lm(ii to ii, jj until nColsInBlock) := 0.0
              ii += 1
              jj += 1
            }
          }
          Iterator.single(((i, j), lm))
        }
      },
      preservesPartitioning = true,
    )

    new BlockMatrix(zeroedBlocks, blockSize, nRows, nCols)
  }

  // for row i, filter to indices [starts[i], stops[i]) by dropping non-overlapping blocks
  // if not blocksOnly, also zero out elements outside ranges in overlapping blocks
  // checked in Python: start >= 0 && start <= stop && stop <= nCols
  def filterRowIntervals(starts: Array[Long], stops: Array[Long], blocksOnly: Boolean)
    : BlockMatrix = {
    require(nRows <= Int.MaxValue)
    require(starts.length == nRows)
    require(stops.length == nRows)

    val filteredBM = filterBlocks(gp.rowIntervalsBlocks(starts, stops))

    if (blocksOnly)
      filteredBM
    else
      filteredBM.zeroRowIntervals(starts, stops)
  }

  def zeroRowIntervals(starts: Array[Long], stops: Array[Long]): BlockMatrix = {
    val backend = HailContext.backend
    val startBlockIndexBc = backend.broadcast(starts.map(gp.indexBlockIndex))
    val stopBlockIndexBc = backend.broadcast(stops.map(stop => (stop / blockSize).toInt))
    val startBlockOffsetBc = backend.broadcast(starts.map(gp.indexBlockOffset))
    val stopBlockOffsetsBc = backend.broadcast(stops.map(gp.indexBlockOffset))

    val zeroedBlocks = blocks.mapPartitions(
      { it =>
        assert(it.hasNext)
        val ((i, j), lm0) = it.next()
        assert(!it.hasNext)

        val lm = lm0.copy // avoidable?

        val startBlockIndex = startBlockIndexBc.value
        val stopBlockIndex = stopBlockIndexBc.value
        val startBlockOffset = startBlockOffsetBc.value
        val stopBlockOffset = stopBlockOffsetsBc.value

        val nRowsInBlock = lm.rows
        val nColsInBlock = lm.cols

        var row = i * blockSize
        var ii = 0
        while (ii < nRowsInBlock) {
          val startBlock = startBlockIndex(row)
          if (startBlock == j)
            lm(ii to ii, 0 until startBlockOffset(row)) := 0.0
          else if (startBlock > j)
            lm(ii to ii, ::) := 0.0
          val stopBlock = stopBlockIndex(row)
          if (stopBlock == j)
            lm(ii to ii, stopBlockOffset(row) until nColsInBlock) := 0.0
          else if (stopBlock < j)
            lm(ii to ii, ::) := 0.0
          row += 1
          ii += 1
        }

        Iterator.single(((i, j), lm))
      },
      preservesPartitioning = true,
    )

    new BlockMatrix(zeroedBlocks, blockSize, nRows, nCols)
  }

  def filterRectangles(flattenedRectangles: Array[Long]): BlockMatrix = {
    require(flattenedRectangles.length % 4 == 0)
    val rectangles = flattenedRectangles.grouped(4).toArray

    filterBlocks(gp.rectanglesBlocks(rectangles))
  }

  def exportRectangles(
    ctx: ExecuteContext,
    output: String,
    rectangles: Array[Array[Long]],
    delimiter: String,
    binary: Boolean,
  ): Unit = {

    val writeRectangleBinary = (uos: OutputStream, dm: BDM[Double]) => {
      val os = new DoubleOutputBuffer(uos, RichArray.defaultBufSize)
      os.writeDoubles(dm.t.toArray)
      os.close()
    }

    val writeRectangleText = (uos: OutputStream, dm: BDM[Double]) => {
      val data = dm.t.toArray
      val nCols = dm.cols
      val os = new OutputStreamWriter(uos)
      val sb = new StringBuilder(blockSize << 2)

      var k = 0
      while (k < data.length - 1) {
        sb.append(data(k))
        if (k % nCols == nCols - 1) {
          sb.append("\n")
        } else {
          sb.append(delimiter)
        }
        k += 1
      }
      sb.append(data.last).append("\n")

      os.write(sb.result())
      os.close()
    }

    val writeRectangle = if (binary) writeRectangleBinary else writeRectangleText

    val dRect = digitsNeeded(rectangles.length)
    val fsBc = ctx.fs.broadcast
    BlockMatrixRectanglesRDD(rectangles, bm = this).foreach { case (index, rectData) =>
      val r = rectangles(index)
      val paddedIndex = StringUtils.leftPad(index.toString, dRect, "0")
      val outputFile = output + "/rect-" + paddedIndex + "_" + r.mkString("-")

      if (rectData.size > 0) {
        using(fsBc.value.create(outputFile))(writeRectangle(_, rectData))
      }
    }
  }

  // element-wise ops
  def unary_+(): M = this

  def unary_-(): M = blockMap(-_, "negation", reqDense = false)

  def add(that: M): M =
    if (sameBlocks(that)) {
      blockMap2(that, _ + _, "addition", reqDense = false)
    } else {
      val addBlocks = new BlockMatrixUnionOpRDD(
        this,
        that,
        _ match {
          case (Some(a), Some(b)) => a + b
          case (Some(a), None) => a
          case (None, Some(b)) => b
          case (None, None) => fatal("not possible for union")
        },
      )
      new BlockMatrix(addBlocks, blockSize, nRows, nCols)
    }

  def sub(that: M): M =
    if (sameBlocks(that)) {
      blockMap2(that, _ - _, "subtraction", reqDense = false)
    } else {
      val subBlocks = new BlockMatrixUnionOpRDD(
        this,
        that,
        _ match {
          case (Some(a), Some(b)) => a - b
          case (Some(a), None) => a
          case (None, Some(b)) => -b
          case (None, None) => fatal("not possible for union")
        },
      )
      new BlockMatrix(subBlocks, blockSize, nRows, nCols)
    }

  def mul(that: M): M = {
    val newGP = gp.intersect(that.gp)
    subsetBlocks(newGP).blockMap2(
      that.subsetBlocks(newGP),
      _ *:* _,
      "element-wise multiplication",
      reqDense = false,
    )
  }

  def div(that: M): M = blockMap2(that, _ /:/ _, "element-wise division")

  // row broadcast
  def rowVectorAdd(a: Array[Double]): M =
    densify().rowVectorOp((lm, lv) => lm(*, ::) + lv, "broadcasted addition of row-vector")(a)

  def rowVectorSub(a: Array[Double]): M =
    densify().rowVectorOp((lm, lv) => lm(*, ::) - lv, "broadcasted subtraction of row-vector")(a)

  def rowVectorMul(a: Array[Double]): M = rowVectorOp(
    (lm, lv) => lm(*, ::) *:* lv,
    "broadcasted multiplication by row-vector containing nan, or infinity",
    reqDense = a.exists(i => i.isNaN | i.isInfinity),
  )(a)

  def rowVectorDiv(a: Array[Double]): M = rowVectorOp(
    (lm, lv) => lm(*, ::) /:/ lv,
    "broadcasted division by row-vector containing zero, nan, or infinity",
    reqDense = a.exists(i => i == 0.0 | i.isNaN | i.isInfinity),
  )(a)

  def reverseRowVectorSub(a: Array[Double]): M = densify().rowVectorOp(
    (lm, lv) => lm(*, ::).map(lv - _),
    "broadcasted row-vector minus block matrix",
  )(a)

  def reverseRowVectorDiv(a: Array[Double]): M = rowVectorOp(
    (lm, lv) => lm(*, ::).map(lv /:/ _),
    "broadcasted row-vector divided by block matrix",
  )(a)

  // column broadcast
  def colVectorAdd(a: Array[Double]): M =
    densify().colVectorOp((lm, lv) => lm(::, *) + lv, "broadcasted addition of column-vector")(a)

  def colVectorSub(a: Array[Double]): M =
    densify().colVectorOp((lm, lv) => lm(::, *) - lv, "broadcasted subtraction of column-vector")(a)

  def colVectorMul(a: Array[Double]): M = colVectorOp(
    (lm, lv) => lm(::, *) *:* lv,
    "broadcasted multiplication column-vector containing nan or infinity",
    reqDense = a.exists(i => i.isNaN | i.isInfinity),
  )(a)

  def colVectorDiv(a: Array[Double]): M = colVectorOp(
    (lm, lv) => lm(::, *) /:/ lv,
    "broadcasted division by column-vector containing zero, nan, or infinity",
    reqDense = a.exists(i => i == 0.0 | i.isNaN | i.isInfinity),
  )(a)

  def reverseColVectorSub(a: Array[Double]): M = densify().colVectorOp(
    (lm, lv) => lm(::, *).map(lv - _),
    "broadcasted column-vector minus block matrix",
  )(a)

  def reverseColVectorDiv(a: Array[Double]): M = colVectorOp(
    (lm, lv) => lm(::, *).map(lv /:/ _),
    "broadcasted column-vector divided by block matrix",
  )(a)

  // scalar
  def scalarAdd(i: Double): M = densify().blockMap(_ + i, "scalar addition")

  def scalarSub(i: Double): M = densify().blockMap(_ - i, "scalar subtraction")

  def scalarMul(i: Double): M =
    blockMap(_ *:* i, s"multiplication by scalar $i", reqDense = i.isNaN | i.isInfinity)

  def scalarDiv(i: Double): M =
    blockMap(_ /:/ i, s"division by scalar $i", reqDense = i == 0.0 | i.isNaN | i.isInfinity)

  def reverseScalarSub(i: Double): M = densify().blockMap(i - _, s"scalar minus block matrix")

  def reverseScalarDiv(i: Double): M = blockMap(i /:/ _, s"scalar divided by block matrix")

  // other element-wise ops
  def sqrt(): M = blockMap(breezeSqrt(_), "sqrt", reqDense = false)

  def ceil(): M = blockMap(breeze.numerics.ceil(_), "ceil", reqDense = false)

  def floor(): M = blockMap(breeze.numerics.floor(_), "floor", reqDense = false)

  def pow(exponent: Double): M = blockMap(
    breezePow(_, exponent),
    s"exponentiation by negative power $exponent",
    reqDense = exponent < 0,
  )

  def log(): M = blockMap(breezeLog(_), "natural logarithm")

  def abs(): M = blockMap(breezeAbs(_), "absolute value", reqDense = false)

  // matrix ops
  def dot(that: M): M =
    new BlockMatrix(new BlockMatrixMultiplyRDD(this, that), blockSize, nRows, that.nCols)

  def dot(lm: BDM[Double]): M = {
    require(
      nCols == lm.rows,
      s"incompatible matrix dimensions: $nRows x $nCols and ${lm.rows} x ${lm.cols}",
    )
    dot(BlockMatrix.fromBreezeMatrix(lm, blockSize))
  }

  def transpose(): M = new BlockMatrix(new BlockMatrixTransposeRDD(this), blockSize, nCols, nRows)

  def sum(): Double = reduce(lm => breezeSum(lm * BDV.ones[Double](lm.cols)), _ + _)

  def rowSum(): BlockMatrix = rowReduce(lm => lm.t * BDV.ones[Double](lm.rows), _ + _)

  def colSum(): BlockMatrix = colReduce(lm => lm * BDV.ones[Double](lm.cols), _ + _)

  def diagonal(): Array[Double] = {
    val nDiagElements = {
      val d = math.min(nRows, nCols)
      if (d > Integer.MAX_VALUE)
        fatal(s"diagonal is too big for local array: $d")
      d.toInt
    }

    val result = new Array[Double](nDiagElements)

    val nDiagBlocks = math.min(gp.nBlockRows, gp.nBlockCols)
    val diagBlocks = Array.tabulate(nDiagBlocks)(i => gp.coordinatesBlock(i, i))

    filterBlocks(diagBlocks).blocks
      .map { case ((i, j), lm) =>
        assert(i == j)
        (i, Array.tabulate(math.min(lm.rows, lm.cols))(ii => lm(ii, ii)))
      }
      .collect()
      .foreach { case (i, a) => System.arraycopy(a, 0, result, i * blockSize, a.length) }

    result
  }

  def write(
    ctx: ExecuteContext,
    uri: String,
    overwrite: Boolean = false,
    forceRowMajor: Boolean = false,
    stageLocally: Boolean = false,
  ): Unit = {
    val fs = ctx.fs
    if (overwrite)
      fs.delete(uri, recursive = true)
    else if (fs.exists(uri))
      fatal(s"file already exists: $uri")

    fs.mkDir(uri)

    def writeBlock(it: Iterator[((Int, Int), BDM[Double])], os: OutputStream): (Long, Long) = {
      assert(it.hasNext)
      val (_, lm) = it.next()
      assert(!it.hasNext)

      val btos = new ByteTrackingOutputStream(os)
      lm.write(btos, forceRowMajor, bufferSpec)
      val bytesWritten = btos.bytesWritten
      btos.close()

      (1L, bytesWritten)
    }

    val fileData = blocks.writePartitions(ctx, uri, stageLocally, writeBlock)

    using(new DataOutputStream(fs.create(uri + metadataRelativePath))) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(
          blockSize,
          nRows,
          nCols,
          gp.partitionIndexToBlockIndex,
          fileData.map(_.path),
        ),
        os,
      )
    }

    using(fs.create(uri + "/_SUCCESS"))(out => ())

    val nBlocks = fileData.length
    assert(nBlocks == fileData.map(_.rowsWritten).sum)
    info(s"wrote matrix with $nRows ${plural(nRows, "row")} " +
      s"and $nCols ${plural(nCols, "column")} " +
      s"as $nBlocks ${plural(nBlocks, "block")} " +
      s"of size $blockSize to $uri")
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
    val level =
      try
        StorageLevel.fromString(storageLevel)
      catch {
        case _: IllegalArgumentException =>
          fatal(s"unknown StorageLevel '$storageLevel'")
      }
    persist(level)
  }

  def unpersist(): this.type = {
    blocks.unpersist()
    this
  }

  def toBreezeMatrix(): BDM[Double] = {
    require(
      nRows <= Int.MaxValue,
      "The number of rows of this matrix should be less than or equal to " +
        s"Int.MaxValue. Currently nRows: $nRows",
    )
    require(
      nCols <= Int.MaxValue,
      "The number of columns of this matrix should be less than or equal to " +
        s"Int.MaxValue. Currently nCols: $nCols",
    )
    require(
      nRows * nCols <= Int.MaxValue,
      "The length of the values array must be " +
        s"less than or equal to Int.MaxValue. Currently nRows * nCols: ${nRows * nCols}",
    )
    val nRowsInt = nRows.toInt
    val nColsInt = nCols.toInt
    val localBlocks = blocks.collect()
    val data = new Array[Double](nRowsInt * nColsInt)
    localBlocks.foreach { case ((i, j), lm) =>
      val iOffset = i * blockSize
      val jOffset = j * blockSize
      var jj = 0
      while (jj < lm.cols) {
        val offset = (jOffset + jj) * nRowsInt + iOffset
        var ii = 0
        while (ii < lm.rows) {
          data(offset + ii) = lm(ii, jj)
          ii += 1
        }
        jj += 1
      }
    }
    new BDM(nRowsInt, nColsInt, data)
  }

  private def requireZippable(that: M, name: String = "operation"): Unit = {
    require(
      nRows == that.nRows,
      s"$name requires same number of rows, but actually: ${nRows}x$nCols, ${that.nRows}x${that.nCols}",
    )
    require(
      nCols == that.nCols,
      s"$name requires same number of cols, but actually: ${nRows}x$nCols, ${that.nRows}x${that.nCols}",
    )
    require(
      blockSize == that.blockSize,
      s"$name requires same block size, but actually: $blockSize and ${that.blockSize}",
    )
    if (!sameBlocks(that))
      fatal(s"$name requires block matrices to have the same set of blocks present")
  }

  private def sameBlocks(that: M): Boolean =
    (gp.partitionIndexToBlockIndex, that.gp.partitionIndexToBlockIndex) match {
      case (Some(bis), Some(bis2)) => bis sameElements bis2
      case (None, None) => true
      case _ => false
    }

  def blockMap(op: BDM[Double] => BDM[Double], name: String = "operation", reqDense: Boolean = true)
    : M = {
    if (reqDense)
      requireDense(name)
    new BlockMatrix(blocks.mapValues(op), blockSize, nRows, nCols)
  }

  def blockMapWithIndex(
    op: ((Int, Int), BDM[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense)
      requireDense(name)
    new BlockMatrix(blocks.mapValuesWithKey(op), blockSize, nRows, nCols)
  }

  def blockMap2(
    that: M,
    op: (BDM[Double], BDM[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks =
      blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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
            val lm = op(lm1, lm2)
            assert(lm.rows == lm1.rows)
            assert(lm.cols == lm1.cols)
            ((i1, j1), lm)
          }
        }
      }
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def map(op: Double => Double, name: String = "operation", reqDense: Boolean = true): M = {
    if (reqDense)
      requireDense(name)
    val newBlocks = blocks.mapValues { lm =>
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def map2(
    that: M,
    op: (Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks =
      blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def map4(
    bm2: M,
    bm3: M,
    bm4: M,
    op: (Double, Double, Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense) {
      requireDense(name)
      bm2.requireDense(name)
      bm3.requireDense(name)
      bm4.requireDense(name)
    }
    requireZippable(bm2)
    requireZippable(bm3)
    requireZippable(bm4)
    val newBlocks =
      blocks.zipPartitions(bm2.blocks, bm3.blocks, bm4.blocks, preservesPartitioning = true) {
        (it1, it2, it3, it4) =>
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
              if (
                lm1.isTranspose == lm2.isTranspose
                && lm1.isTranspose == lm3.isTranspose
                && lm1.isTranspose == lm4.isTranspose
              ) {
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def mapWithIndex(
    op: (Long, Long, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense)
      requireDense(name)
    val newBlocks = blocks.mapValuesWithKey { case ((i, j), lm) =>
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def map2WithIndex(
    that: M,
    op: (Long, Long, Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true,
  ): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks =
      blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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
                result(ii + jj * lm1.rows) =
                  op(iOffset + ii, jOffset + jj, lm1(ii, jj), lm2(ii, jj))
                ii += 1
              }
              jj += 1
            }
            ((i1, j1), new BDM(lm1.rows, lm1.cols, result))
          }
        }
      }
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def colVectorOp(
    op: (BDM[Double], BDV[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true,
  ): Array[Double] => M = {
    a =>
      val v = BDV(a)
      require(v.length == nRows, s"vector length must equal nRows: ${v.length}, $nRows")
      val vBc = HailContext.backend.broadcast(v)
      blockMapWithIndex(
        { case ((i, _), lm) =>
          val lv = gp.vectorOnBlockRow(vBc.value, i)
          op(lm, lv)
        },
        name,
        reqDense = reqDense,
      )
  }

  def rowVectorOp(
    op: (BDM[Double], BDV[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true,
  ): Array[Double] => M = {
    a =>
      val v = BDV(a)
      require(v.length == nCols, s"vector length must equal nCols: ${v.length}, $nCols")
      val vBc = HailContext.backend.broadcast(v)
      blockMapWithIndex(
        { case ((_, j), lm) =>
          val lv = gp.vectorOnBlockCol(vBc.value, j)
          op(lm, lv)
        },
        name,
        reqDense = reqDense,
      )
  }

  def reduce(blockOp: BDM[Double] => Double, scalarOp: (Double, Double) => Double): Double =
    blocks
      .map { case ((_, _), lm) => blockOp(lm) }
      .fold(0.0)(scalarOp)

  def rowReduce(
    blockOp: BDM[Double] => BDV[Double],
    vectorOp: (BDV[Double], BDV[Double]) => BDV[Double],
  ): BlockMatrix =
    new BlockMatrix(
      blocks
        .map { case ((_, j), lm) => ((0, j), blockOp(lm)) }
        .reduceByKey(GridPartitioner(blockSize, 1, nCols, gp.maybeBlockCols()), vectorOp)
        .mapValues(v => new BDM[Double](1, v.length, v.data)),
      blockSize,
      1,
      nCols,
    )

  def colReduce(
    blockOp: BDM[Double] => BDV[Double],
    vectorOp: (BDV[Double], BDV[Double]) => BDV[Double],
  ): BlockMatrix =
    new BlockMatrix(
      blocks
        .map { case ((i, _), lm) => ((i, 0), blockOp(lm)) }
        .reduceByKey(GridPartitioner(blockSize, nRows, 1, gp.maybeBlockRows()), vectorOp)
        .mapValues(v => new BDM[Double](v.length, 1, v.data)),
      blockSize,
      nRows,
      1,
    )

  def toIndexedRowMatrix(): IndexedRowMatrix = {
    require(nCols <= Integer.MAX_VALUE)
    val nColsInt = nCols.toInt

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

    new IndexedRowMatrix(
      blocks.flatMap { case ((i, j), lm) =>
        val iOffset = i * blockSize
        val jOffset = j * blockSize

        for (k <- 0 until lm.rows)
          yield (k + iOffset, (jOffset, lm(k, ::).inner.toArray))
      }.aggregateByKey(new Array[Double](nColsInt))(seqOp, combOp)
        .map { case (i, a) => IndexedRow(i, BDV(a)) },
      nRows,
      nColsInt,
    )
  }

  def getElement(row: Long, col: Long): Double = {
    val blockRow = gp.indexBlockIndex(row)
    val blockCol = gp.indexBlockIndex(col)
    val pi = gp.coordinatesPart(blockRow, blockCol)
    if (pi >= 0) {
      val rowOffset = gp.indexBlockOffset(row)
      val colOffset = gp.indexBlockOffset(col)
      blocks.subsetPartitions(Array(pi))
        .map { case ((i, j), lm) =>
          assert(i == blockRow && j == blockCol)
          lm(rowOffset, colOffset)
        }
        .collect()(0)
    } else
      0.0
  }

  def filterRows(keep: Array[Long]): BlockMatrix =
    new BlockMatrix(new BlockMatrixFilterRowsRDD(this, keep), blockSize, keep.length, nCols)

  def filterCols(keep: Array[Long]): BlockMatrix =
    new BlockMatrix(new BlockMatrixFilterColsRDD(this, keep), blockSize, nRows, keep.length)

  def filter(keepRows: Array[Long], keepCols: Array[Long]): BlockMatrix =
    new BlockMatrix(
      new BlockMatrixFilterRDD(this, keepRows, keepCols),
      blockSize,
      keepRows.length,
      keepCols.length,
    )

  def entriesTable(ctx: ExecuteContext): TableValue = {
    val rowType = PCanonicalStruct(
      true,
      "i" -> PInt64Required,
      "j" -> PInt64Required,
      "entry" -> PFloat64Required,
    )

    val sm = ctx.stateManager
    val entriesRDD =
      ContextRDD.weaken(blocks).cflatMap { case (rvdContext, ((blockRow, blockCol), block)) =>
        val rowOffset = blockRow * blockSize.toLong
        val colOffset = blockCol * blockSize.toLong

        val rvb = new RegionValueBuilder(sm, rvdContext.region)

        block.activeIterator
          .map { case ((i, j), entry) =>
            rvb.start(rowType)
            rvb.startStruct()
            rvb.addLong(rowOffset + i)
            rvb.addLong(colOffset + j)
            rvb.addDouble(entry)
            rvb.endStruct()
            rvb.end()
          }
      }

    TableValue(ctx, rowType, FastSeq(), entriesRDD)
  }
}

case class BlockMatrixFilterRDDPartition(
  index: Int,
  blockRowRanges: Array[(Int, Array[Int], Array[Int])],
  blockColRanges: Array[(Int, Array[Int], Array[Int])],
) extends Partition

object BlockMatrixFilterRDD {
  /* allBlockColRanges(newBlockCol) has elements of the form (blockCol, startIndices, endIndices)
   * with blockCol increasing */
  /* startIndices.zip(endIndices) gives all column-index ranges in blockCol to be copied to ranges
   * in newBlockCol */
  def computeAllBlockColRanges(keep: Array[Long], gp: GridPartitioner, newGP: GridPartitioner)
    : Array[Array[(Int, Array[Int], Array[Int])]] = {

    val blockSize = gp.blockSize
    val ab = new BoxedArrayBuilder[(Int, Array[Int], Array[Int])]()
    val startIndices = new IntArrayBuilder()
    val endIndices = new IntArrayBuilder()

    keep
      .grouped(blockSize)
      .zipWithIndex
      .map { case (colsInNewBlock, newBlockCol) =>
        ab.clear()

        val newBlockNCols = newGP.blockColNCols(newBlockCol)

        var j = 0 // start index in newBlockCol
        while (j < newBlockNCols) {
          startIndices.clear()
          endIndices.clear()

          val startCol = colsInNewBlock(j)
          val blockCol = (startCol / blockSize).toInt
          val finalColInBlockCol = blockCol * blockSize + gp.blockColNCols(blockCol)

          while (j < newBlockNCols && colsInNewBlock(j) < finalColInBlockCol) { // compute ranges for this blockCol
            val startCol = colsInNewBlock(j)
            val startColIndex = (startCol % blockSize).toInt // start index in blockCol
            startIndices += startColIndex

            var endCol = startCol + 1
            var k = j + 1
            while (
              k < newBlockNCols && colsInNewBlock(k) == endCol && endCol < finalColInBlockCol
            ) { // extend range
              endCol += 1
              k += 1
            }
            endIndices += ((endCol - 1) % blockSize + 1).toInt // end index in blockCol
            j = k
          }
          ab += ((blockCol, startIndices.result(), endIndices.result()))
        }
        ab.result()
      }.toArray
  }

  def computeAllBlockRowRanges(keep: Array[Long], gp: GridPartitioner, newGP: GridPartitioner)
    : Array[Array[(Int, Array[Int], Array[Int])]] =
    computeAllBlockColRanges(keep, gp.transpose._1, newGP.transpose._1)
}

// checked in Python: keepRows and keepCols non-empty, increasing, valid range
private class BlockMatrixFilterRDD(bm: BlockMatrix, keepRows: Array[Long], keepCols: Array[Long])
    extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {
  log.info("Constructing BlockMatrixFilterRDD")

  val t0 = System.nanoTime()

  private val originalGP = bm.gp

  if (bm.isSparse) {
    log.info("Filtering a sparse matrix")
  }

  private val blockSize = originalGP.blockSize
  @transient private val tempDenseGP = GridPartitioner(blockSize, keepRows.length, keepCols.length)

  private val allBlockRowRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockRowRanges(keepRows, originalGP, tempDenseGP)

  private val allBlockColRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockColRanges(keepCols, originalGP, tempDenseGP)

  private val originalMaybeBlocksSet = originalGP.partitionIndexToBlockIndex.map(_.toSet)

  private val blockParentMap = (0 until tempDenseGP.numPartitions).map { blockId =>
    val (newBlockRow, newBlockCol) = tempDenseGP.blockCoordinates(blockId)

    val parents = for {
      blockRow <- allBlockRowRanges(newBlockRow).map(_._1)
      blockCol <- allBlockColRanges(newBlockCol).map(_._1)
    } yield originalGP.coordinatesBlock(blockRow, blockCol)
    (blockId, parents)
  }.map { case (blockId, parents) =>
    val filteredParents = originalMaybeBlocksSet match {
      case None => parents
      case Some(blockIdSet) => parents.filter(id => blockIdSet.contains(id))
    }
    (blockId, filteredParents)
  }.filter { case (_, parents) => !parents.isEmpty }.toMap

  private val blockIndices = blockParentMap.keys.toArray.sorted

  private val newGPMaybeBlocks: Option[IndexedSeq[Int]] =
    if (blockIndices.length == tempDenseGP.maxNBlocks) None else Some(blockIndices)

  private val newGP = tempDenseGP.copy(partitionIndexToBlockIndex = newGPMaybeBlocks)

  log.info(
    s"Finished constructing block matrix filter RDD. Total time ${(System.nanoTime() - t0).toDouble / 1000000000}"
  )

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { partitionIndex =>
      val blockIndex = newGP.partitionToBlock(partitionIndex)
      BlockMatrixFilterRDDPartition(
        partitionIndex,
        allBlockRowRanges(newGP.blockBlockRow(blockIndex)),
        allBlockColRanges(newGP.blockBlockCol(blockIndex)),
      )
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val blockForPartition = newGP.partitionToBlock(partitionId)
        val blockParents = blockParentMap(blockForPartition)
        val partitionParents =
          blockParents.map(blockId => originalGP.blockToPartition(blockId)).toSet.toArray.sorted
        partitionParents
      }
    }
  )

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val part = split.asInstanceOf[BlockMatrixFilterRDDPartition]

    val blockForPartition = newGP.partitionToBlock(split.index)
    val (newBlockRow, newBlockCol) = newGP.blockCoordinates(blockForPartition)
    val (newBlockNRows, newBlockNCols) = newGP.blockDims(blockForPartition)
    val parentZeroBlock = BDM.zeros[Double](originalGP.blockSize, originalGP.blockSize)
    val newBlock = BDM.zeros[Double](newBlockNRows, newBlockNCols)

    log.info(s"Computing partition for FilterRDD $part")

    var jCol = 0
    var kCol = 0
    part.blockColRanges.foreach { case (blockCol, colStartIndices, colEndIndices) =>
      val jCol0 = jCol // record first col index in newBlock corresponding to new blockCol
      var jRow = 0
      var kRow = 0
      part.blockRowRanges.foreach { case (blockRow, rowStartIndices, rowEndIndices) =>
        val jRow0 = jRow // record first row index in newBlock corresponding to new blockRow

        val parentBI = originalGP.coordinatesBlock(blockRow, blockCol)
        var block = parentZeroBlock

        if (blockParentMap(newGP.partitionToBlock(split.index)).contains(parentBI)) {
          val parentPI = originalGP.blockToPartition(parentBI)
          block = bm.blocks.iterator(bm.blocks.partitions(parentPI), context).next()._2
        }

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

            newBlock(jRow until kRow, jCol until kCol) := block(
              siRow until eiRow,
              siCol until eiCol,
            )

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

case class BlockMatrixFilterOneDimRDDPartition(
  index: Int,
  blockRanges: Array[(Int, Array[Int], Array[Int])],
) extends Partition

// checked in Python: keep non-empty, increasing, valid range
private class BlockMatrixFilterColsRDD(bm: BlockMatrix, keep: Array[Long])
    extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {

  private val childPartitionsBc = bm.blocks.sparkContext.broadcast(bm.blocks.partitions)

  private val originalGP = bm.gp
  private val blockSize = originalGP.blockSize
  @transient private val tempDenseGP = GridPartitioner(blockSize, originalGP.nRows, keep.length)

  @transient private val allBlockColRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockColRanges(keep, originalGP, tempDenseGP)

  @transient private val originalMaybeBlocksSet = originalGP.partitionIndexToBlockIndex.map(_.toSet)

  /* Map the denseGP blocks to the blocks of parents they depend on, temporarily pretending they are
   * all there. */
  // Then delete the parents that aren't in originalGP.maybeBlocks, then delete the pairs
  // without parents at all.
  @transient private val blockParentMap = (0 until tempDenseGP.numPartitions).map { blockId =>
    val (blockRow, newBlockCol) = tempDenseGP.blockCoordinates(blockId)
    blockId -> allBlockColRanges(newBlockCol).map { case (blockCol, _, _) =>
      originalGP.coordinatesBlock(blockRow, blockCol)
    }
  }.map { case (blockId, parents) =>
    val filteredParents = originalMaybeBlocksSet match {
      case None => parents
      case Some(blockIdSet) => parents.filter(id => blockIdSet.contains(id))
    }
    (blockId, filteredParents)
  }.filter { case (_, parents) => !parents.isEmpty }.toMap

  private val blockParentMapBc = bm.blocks.sparkContext.broadcast(blockParentMap)

  @transient private val blockIndices = blockParentMap.keys.toFastSeq.sorted

  @transient private val newGPMaybeBlocks =
    if (blockIndices.length == tempDenseGP.maxNBlocks) None else Some(blockIndices)

  private val newGP = tempDenseGP.copy(partitionIndexToBlockIndex = newGPMaybeBlocks)

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { partitionIndex: Int =>
      val blockIndex = newGP.partitionToBlock(partitionIndex)
      BlockMatrixFilterOneDimRDDPartition(
        partitionIndex,
        allBlockColRanges(newGP.blockBlockCol(blockIndex)),
      )
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val blockForPartition = newGP.partitionToBlock(partitionId)
        val blockParents = blockParentMap(blockForPartition)
        val partitionParents =
          blockParents.map(blockId => originalGP.blockToPartition(blockId)).toSet.toArray.sorted
        partitionParents
      }
    }
  )

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val blockIndex = newGP.partitionToBlock(split.index)
    val (blockRow, newBlockCol) = newGP.blockCoordinates(blockIndex)
    val (blockNRows, newBlockNCols) = newGP.blockDims(blockIndex)
    val parentZeroBlock = BDM.zeros[Double](originalGP.blockSize, originalGP.blockSize)
    val newBlock = BDM.zeros[Double](blockNRows, newBlockNCols)
    var j = 0
    var k = 0

    val splitCast = split.asInstanceOf[BlockMatrixFilterOneDimRDDPartition]

    splitCast
      .blockRanges
      .foreach { case (blockCol, startIndices, endIndices) =>
        val parentBI = originalGP.coordinatesBlock(blockRow, blockCol)
        var block = parentZeroBlock

        if (blockParentMapBc.value(newGP.partitionToBlock(split.index)).contains(parentBI)) {
          val parentPI = originalGP.blockToPartition(parentBI)
          block = bm.blocks.iterator(childPartitionsBc.value(parentPI), context).next()._2
        }
        var colRangeIndex = 0
        while (colRangeIndex < startIndices.length) {
          val si = startIndices(colRangeIndex)
          val ei = endIndices(colRangeIndex)
          k = j + ei - si

          newBlock(::, j until k) := block(0 until newBlock.rows, si until ei)

          j = k
          colRangeIndex += 1
        }
      }
    assert(j == newBlockNCols)

    Iterator.single(((blockRow, newBlockCol), newBlock))
  }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

// checked in Python: keep non-empty, increasing, valid range
private class BlockMatrixFilterRowsRDD(bm: BlockMatrix, keep: Array[Long])
    extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {

  private val childPartitionsBc = bm.blocks.sparkContext.broadcast(bm.blocks.partitions)

  private val originalGP = bm.gp
  private val blockSize = originalGP.blockSize
  private val tempDenseGP = GridPartitioner(blockSize, keep.length, originalGP.nCols)

  @transient private val allBlockRowRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockRowRanges(keep, originalGP, tempDenseGP)

  @transient private val originalMaybeBlocksSet = originalGP.partitionIndexToBlockIndex.map(_.toSet)

  /* Map the denseGP blocks to the blocks of parents they depend on, temporarily pretending they are
   * all there. */
  // Then delete the parents that aren't in originalGP.maybeBlocks, then delete the pairs
  // without parents at all.
  @transient private val blockParentMap = (0 until tempDenseGP.numPartitions).map { blockId =>
    val (newBlockRow, blockCol) = tempDenseGP.blockCoordinates(blockId)
    blockId -> allBlockRowRanges(newBlockRow).map { case (blockRow, _, _) =>
      originalGP.coordinatesBlock(blockRow, blockCol)
    }
  }.map { case (blockId, parents) =>
    val filteredParents = originalMaybeBlocksSet match {
      case None => parents
      case Some(blockIdSet) => parents.filter(id => blockIdSet.contains(id))
    }
    (blockId, filteredParents)
  }.filter { case (_, parents) => !parents.isEmpty }.toMap

  private val blockParentMapBc = bm.blocks.sparkContext.broadcast(blockParentMap)

  @transient private val blockIndices = blockParentMap.keys.toFastSeq.sorted

  @transient private val newGPMaybeBlocks =
    if (blockIndices.length == tempDenseGP.maxNBlocks) None else Some(blockIndices)

  private val newGP = tempDenseGP.copy(partitionIndexToBlockIndex = newGPMaybeBlocks)

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { partitionIndex: Int =>
      val blockIndex = newGP.partitionToBlock(partitionIndex)
      BlockMatrixFilterOneDimRDDPartition(
        partitionIndex,
        allBlockRowRanges(newGP.blockBlockRow(blockIndex)),
      )
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val blockForPartition = newGP.partitionToBlock(partitionId)
        val blockParents = blockParentMap(blockForPartition)
        val partitionParents =
          blockParents.map(blockId => originalGP.blockToPartition(blockId)).toSet.toArray.sorted
        partitionParents
      }
    }
  )

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val blockIndex = newGP.partitionToBlock(split.index)
    val (newBlockRow, blockCol) = newGP.blockCoordinates(blockIndex)
    val (newBlockNRows, blockNCols) = newGP.blockDims(blockIndex)
    val parentZeroBlock = BDM.zeros[Double](originalGP.blockSize, originalGP.blockSize)
    val newBlock = BDM.zeros[Double](newBlockNRows, blockNCols)
    var j = 0
    var k = 0

    val splitCast = split.asInstanceOf[BlockMatrixFilterOneDimRDDPartition]

    splitCast
      .blockRanges
      .foreach { case (blockRow, startIndices, endIndices) =>
        val parentBI = originalGP.coordinatesBlock(blockRow, blockCol)
        var block = parentZeroBlock

        if (blockParentMapBc.value(newGP.partitionToBlock(split.index)).contains(parentBI)) {
          val parentPI = originalGP.blockToPartition(parentBI)
          block = bm.blocks.iterator(childPartitionsBc.value(parentPI), context).next()._2
        }
        var rowRangeIndex = 0
        while (rowRangeIndex < startIndices.length) {
          val si = startIndices(rowRangeIndex)
          val ei = endIndices(rowRangeIndex)
          k = j + ei - si

          newBlock(j until k, ::) := block(si until ei, 0 until newBlock.cols)

          j = k
          rowRangeIndex += 1
        }
      }
    assert(j == newBlockNRows)

    Iterator.single(((newBlockRow, blockCol), newBlock))
  }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

case class BlockMatrixTransposeRDDPartition(index: Int, prevPartition: Partition) extends Partition

private class BlockMatrixTransposeRDD(bm: BlockMatrix)
    extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {

  private val (newGP, transposedPartitionIndicesToParentPartitions) = bm.gp.transpose

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = {
        val parent = transposedPartitionIndicesToParentPartitions(partitionId)
        val (oldI, oldJ) = bm.gp.partCoordinates(parent)
        val (newI, newJ) = newGP.partCoordinates(partitionId)
        assert(newI == oldJ && newJ == oldI)
        Array(parent)
      }
    }
  )

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
    bm.blocks.iterator(split.asInstanceOf[BlockMatrixTransposeRDDPartition].prevPartition, context)
      .map { case ((j, i), lm) => ((i, j), lm.t) }

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { pi =>
      BlockMatrixTransposeRDDPartition(
        pi,
        bm.blocks.partitions(transposedPartitionIndicesToParentPartitions(pi)),
      )
    }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

private class BlockMatrixUnionOpRDD(
  l: BlockMatrix,
  r: BlockMatrix,
  op: ((Option[BDM[Double]], Option[BDM[Double]])) => BDM[Double],
) extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {

  import BlockMatrix.block

  require(l.blockSize == r.blockSize)
  require(l.nRows == r.nRows)
  require(l.nCols == r.nCols)

  private val lGP = l.gp
  private val rGP = r.gp
  private val gp = lGP.union(rGP)

  private val lParts = l.blocks.partitions
  private val rParts = r.blocks.partitions

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] =
          Array(lGP.blockToPartition(gp.partitionToBlock(partitionId))).filter(_ >= 0)
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] =
          Array(rGP.blockToPartition(gp.partitionToBlock(partitionId))).filter(_ >= 0)
      },
    )

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (i, j) = gp.partCoordinates(split.index)
    val lm = op(block(l, lParts, lGP, context, i, j) -> block(r, rParts, rGP, context, i, j))

    Iterator.single(((i, j), lm))
  }

  protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
    new Partition { def index: Int = pi }
  )

  @transient override val partitioner: Option[Partitioner] = Some(gp)
}

private class BlockMatrixMultiplyRDD(l: BlockMatrix, r: BlockMatrix)
    extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {

  import BlockMatrix.block

  require(
    l.nCols == r.nRows,
    s"inner dimensions must match, but given: ${l.nRows}x${l.nCols}, ${r.nRows}x${r.nCols}",
  )

  require(
    l.blockSize == r.blockSize,
    s"blocks must be same size, but actually were ${l.blockSize}x${l.blockSize} and ${r.blockSize}x${r.blockSize}",
  )

  private val lGP = l.gp
  private val rGP = r.gp
  private val gp = GridPartitioner(l.blockSize, l.nRows, r.nCols)

  private val lParts = l.blocks.partitions
  private val rParts = r.blocks.partitions
  private val nProducts = lGP.nBlockCols

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(l.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val i = gp.blockBlockRow(partitionId)
          (0 until nProducts).map(k => lGP.coordinatesPart(i, k)).filter(_ >= 0).toArray
        }
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] = {
          val j = gp.blockBlockCol(partitionId)
          (0 until nProducts).map(k => rGP.coordinatesPart(k, j)).filter(_ >= 0).toArray
        }
      },
    )

  def fma(c: BDM[Double], _a: BDM[Double], _b: BDM[Double]): Unit = {
    assert(_a.cols == _b.rows)

    val a =
      if (_a.majorStride < math.max(if (_a.isTranspose) _a.cols else _a.rows, 1)) _a.copy else _a
    val b =
      if (_b.majorStride < math.max(if (_b.isTranspose) _b.cols else _b.rows, 1)) _b.copy else _b

    import com.github.fommil.netlib.BLAS.{getInstance => blas}
    blas.dgemm(
      if (a.isTranspose) "T" else "N",
      if (b.isTranspose) "T" else "N",
      c.rows,
      c.cols,
      a.cols,
      1.0,
      a.data,
      a.offset,
      a.majorStride,
      b.data,
      b.offset,
      b.majorStride,
      1.0,
      c.data,
      0,
      c.rows,
    )
  }

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (i, j) = gp.blockCoordinates(split.index)
    val (blockNRows, blockNCols) = gp.blockDims(split.index)
    val product = BDM.zeros[Double](blockNRows, blockNCols)
    var k = 0
    while (k < nProducts) {
      val left = block(l, lParts, lGP, context, i, k)
      val right = block(r, rParts, rGP, context, k, j)
      if (left.isDefined && right.isDefined) {
        fma(product, left.get, right.get)
      }
      k += 1
    }
    Iterator.single(((i, j), product))
  }

  protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
    new Partition { def index: Int = pi }
  )

  @transient override val partitioner: Option[Partitioner] = Some(gp)
}

case class BlockMatrixRectanglesRDD(rectangles: Array[Array[Long]], bm: BlockMatrix)
    extends RDD[(Int, BDM[Double])](bm.blocks.sparkContext, Nil) {

  assert(rectangles.forall(rect => rect.length == 4))

  assert(rectangles.forall(rect =>
    rect(1) - rect(0) <= Int.MaxValue && rect(3) - rect(2) <= Int.MaxValue
  ))

  val gp: GridPartitioner = bm.gp

  override def compute(split: Partition, context: TaskContext): Iterator[(Int, BDM[Double])] = {
    val rect = rectangles(split.index)
    val Array(rectStartRow, rectEndRow, rectStartCol, rectEndCol) = rect

    val rectData =
      new BDM[Double]((rectEndRow - rectStartRow).toInt, (rectEndCol - rectStartCol).toInt)
    val blocksInRectangle = gp.rectangleBlocks(rect)
    blocksInRectangle.foreach { blockIdx =>
      val (blockRowIdx, blockColIdx) = gp.blockCoordinates(blockIdx)
      val blockStartRow = blockRowIdx * gp.blockSize
      val blockStartCol = blockColIdx * gp.blockSize
      val blockEndRow = blockStartRow + gp.blockRowNRows(blockRowIdx)
      val blockEndCol = blockStartCol + gp.blockColNCols(blockColIdx)

      val blockRowSlice = overlapBlockSlice(rectStartRow, rectEndRow, blockStartRow, blockEndRow)
      val blockColSlice = overlapBlockSlice(rectStartCol, rectEndCol, blockStartCol, blockEndCol)
      val rectRowSlice = overlapRectSlice(rectStartRow, rectEndRow, blockStartRow, blockEndRow)
      val rectColSlice = overlapRectSlice(rectStartCol, rectEndCol, blockStartCol, blockEndCol)

      BlockMatrix.block(bm, bm.blocks.partitions, gp, context, blockRowIdx, blockColIdx).foreach {
        block => rectData(rectRowSlice, rectColSlice) := block(blockRowSlice, blockColSlice)
      }
    }

    Iterator.single((split.index, rectData))
  }

  private def overlapBlockSlice(rectStart: Long, rectEnd: Long, blockStart: Long, blockEnd: Long)
    : Range = {
    val (start, end) = absoluteOverlap(rectStart, rectEnd, blockStart, blockEnd)
    (start - blockStart).toInt until (end - blockStart).toInt
  }

  private def overlapRectSlice(rectStart: Long, rectEnd: Long, blockStart: Long, blockEnd: Long)
    : Range = {
    val (start, end) = absoluteOverlap(rectStart, rectEnd, blockStart, blockEnd)
    (start - rectStart).toInt until (end - rectStart).toInt
  }

  private def absoluteOverlap(rectStart: Long, rectEnd: Long, blockStart: Long, blockEnd: Long)
    : (Long, Long) =
    (Math.max(rectStart, blockStart), Math.min(rectEnd, blockEnd))

  override protected def getPartitions: Array[Partition] =
    Array.tabulate(rectangles.length)(rectIndex => new Partition { val index: Int = rectIndex })
}

// On compute, WriteBlocksRDDPartition writes the block row with index `index`
// [`start`, `end`] is the range of indices of parent partitions overlapping this block row
// `skip` is the index in the start partition corresponding to the first row of this block row
case class WriteBlocksRDDPartition(
  index: Int,
  start: Int,
  skip: Int,
  end: Int,
  parentPartitions: Array[Partition],
) extends Partition {
  def range: Range = start to end
}

class WriteBlocksRDD(
  fsBc: BroadcastValue[FS],
  localTmpDir: String,
  path: String,
  rvd: RVD,
  parentPartStarts: Array[Long],
  entryField: String,
  gp: GridPartitioner,
) extends RDD[(Int, String)](SparkBackend.sparkContext("WriteBlocksRDD"), Nil) {

  require(gp.nRows == parentPartStarts.last)

  private val blockSize = gp.blockSize
  private val crdd = rvd.crdd
  private val rvRowType = rvd.rowPType

  private val d = digitsNeeded(gp.numPartitions)

  override def getDependencies: Seq[Dependency[_]] =
    Array[Dependency[_]](
      new NarrowDependency(crdd.rdd) {
        def getParents(partitionId: Int): Seq[Int] =
          partitions(partitionId).asInstanceOf[WriteBlocksRDDPartition].range
      }
    )

  protected def getPartitions: Array[Partition] = {
    val nRows = parentPartStarts.last
    assert(nRows == gp.nRows)
    val nBlockRows = gp.nBlockRows

    val parts = new Array[Partition](nBlockRows)
    val parentPartitions = crdd.partitions

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

      parts(blockRow) = WriteBlocksRDDPartition(
        blockRow,
        start,
        skip,
        end,
        (start to end).map(i => parentPartitions(i)).toArray,
      )

      firstRowInBlock = firstRowInNextBlock
      blockRow += 1
    }

    parts
  }

  def compute(split: Partition, context: TaskContext): Iterator[(Int, String)] = {
    val blockRow = split.index
    val nRowsInBlock = gp.blockRowNRows(blockRow)
    val ctx = TaskContext.get

    val (blockPartFiles, outPerBlockCol, paths) = Array.tabulate(gp.nBlockCols) { blockCol =>
      val nColsInBlock = gp.blockColNCols(blockCol)

      val i = gp.coordinatesBlock(blockRow, blockCol)
      val f = partFile(d, i, ctx)

      val finalPath = path + "/parts/" + f
      val tmpPath = ExecuteContext.createTmpPathNoCleanup(localTmpDir, "writeBlocksRDD")

      val os = fsBc.value.create(tmpPath)
      val out = BlockMatrix.bufferSpec.buildOutputBuffer(os)

      out.writeInt(nRowsInBlock)
      out.writeInt(nColsInBlock)
      out.writeBoolean(true) // transposed, stored row major

      ((i, f), out, (tmpPath, finalPath))
    }.unzip3

    val entryArrayType = MatrixType.getEntryArrayType(rvRowType)
    val entryType = MatrixType.getEntryType(rvRowType)
    val fieldType = entryType.field(entryField).typ

    assert(fieldType.virtualType == TFloat64)

    val entryArrayIdx = MatrixType.getEntriesIndex(rvRowType)
    val fieldIdx = entryType.fieldIdx(entryField)

    val data = new Array[Double](blockSize)
    val writeBlocksPart = split.asInstanceOf[WriteBlocksRDDPartition]
    val start = writeBlocksPart.start
    writeBlocksPart.range.zip(writeBlocksPart.parentPartitions).foreach { case (pi, pPart) =>
      using(RVDContext.default(SparkTaskContext.get().getRegionPool())) { ctx =>
        val it = crdd.iterator(pPart, context, ctx)

        if (pi == start) {
          var j = 0
          while (j < writeBlocksPart.skip) {
            it.next()
            ctx.region.clear()
            j += 1
          }
        }

        var i = 0
        while (it.hasNext && i < nRowsInBlock) {
          val rv = it.next()

          val entryArrayOffset = rvRowType.loadField(rv, entryArrayIdx)

          var blockCol = 0
          var colIdx = 0
          val colIt = entryArrayType.elementIterator(
            entryArrayOffset,
            entryArrayType.loadLength(entryArrayOffset),
          )
          while (blockCol < gp.nBlockCols) {
            val n = gp.blockColNCols(blockCol)
            var j = 0
            while (j < n) {
              assert(colIt.hasNext)
              if (colIt.isDefined) {
                val entryOffset = colIt.value
                if (entryType.isFieldDefined(entryOffset, fieldIdx)) {
                  val fieldOffset = entryType.loadField(entryOffset, fieldIdx)
                  data(j) = Region.loadDouble(fieldOffset)
                } else {
                  val rowIdx = blockRow * blockSize + i
                  fatal(s"Cannot create BlockMatrix: missing value at row $rowIdx and col $colIdx")
                }
              } else {
                val rowIdx = blockRow * blockSize + i
                fatal(s"Cannot create BlockMatrix: filtered entry at row $rowIdx and col $colIdx")
              }
              colIdx += 1
              colIt.iterate()
              j += 1
            }
            outPerBlockCol(blockCol).writeDoubles(data, 0, n)
            blockCol += 1
          }
          i += 1
          ctx.region.clear()
        }
      }
    }
    outPerBlockCol.foreach(_.close())
    paths.foreach { case (tempPath, finalPath) =>
      fsBc.value.copy(tempPath, finalPath, deleteSource = true)
    }
    blockPartFiles.iterator
  }
}

object BlockMatrixReadRowBlockedRDD {
  val DEFAULT_MAXIMUM_CACHE_MEMORY_IN_BYTES = 32 * 1024 * 1024
}

class BlockMatrixReadRowBlockedRDD(
  fsBc: BroadcastValue[FS],
  path: String,
  partitionRanges: IndexedSeq[NumericRange.Exclusive[Long]],
  requestedType: TStruct,
  metadata: BlockMatrixMetadata,
  maybeMaximumCacheMemoryInBytes: Option[Int],
) extends RDD[RVDContext => Iterator[Long]](
      SparkBackend.sparkContext("BlockMatrixReadRowBlockedRDD"),
      Nil,
    ) {
  import BlockMatrixReadRowBlockedRDD._

  private[this] val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) =
    metadata

  private[this] val gp = GridPartitioner(blockSize, nRows, nCols)

  private[this] val maximumCacheMemoryInBytes =
    maybeMaximumCacheMemoryInBytes.getOrElse(DEFAULT_MAXIMUM_CACHE_MEMORY_IN_BYTES)

  private[this] val doublesPerFile = maximumCacheMemoryInBytes / (gp.nBlockCols * 8)

  assert(
    doublesPerFile >= blockSize,
    "BlockMatrixCachedPartFile must be able to hold at least one row of every block in memory",
  )

  override def compute(split: Partition, context: TaskContext)
    : Iterator[RVDContext => Iterator[Long]] = {
    val pi = split.index
    val rowsForPartition = partitionRanges(pi)
    val createRowIdx = requestedType.fieldNames.contains("row_idx")
    val createRowUID = requestedType.fieldNames.contains(TableReader.uidFieldName)
    assert(requestedType.fieldNames.contains("entries"))
    val rowPType = PCanonicalStruct(
      Array(
        if (createRowIdx) Some("row_idx" -> PInt64()) else None,
        Some("entries" -> PCanonicalArray(PFloat64())),
        if (createRowUID) Some(TableReader.uidFieldName -> PInt64()) else None,
      ).flatten: _*
    )

    if (rowsForPartition.isEmpty) {
      return Iterator.single(ctx => Iterator.empty)
    }
    Iterator.single { ctx =>
      val region = ctx.region
      val rvb = new RegionValueBuilder(HailStateManager(Map.empty), region)
      val rv = RegionValue(region)
      val firstRow = rowsForPartition(0)
      var blockRow = (firstRow / blockSize).toInt
      val fs = fsBc.value
      var pfs = Array.tabulate(gp.nBlockCols) { blockCol =>
        new BlockMatrixCachedPartFile(
          (firstRow % blockSize).toInt,
          doublesPerFile,
          fs,
          path,
          partFiles(gp.coordinatesBlock(blockRow, blockCol)),
        )
      }

      rowsForPartition.iterator.map { row =>
        val nextBlockRow = (row / blockSize).toInt
        if (nextBlockRow != blockRow) {
          assert(row % blockSize == 0)
          blockRow = nextBlockRow
          pfs = Array.tabulate(gp.nBlockCols) { blockCol =>
            new BlockMatrixCachedPartFile(
              0,
              doublesPerFile,
              fs,
              path,
              partFiles(gp.coordinatesBlock(blockRow, blockCol)),
            )
          }
        }

        rvb.start(rowPType)
        rvb.startStruct()
        if (createRowIdx) rvb.addLong(row)
        assert(nCols < Int.MaxValue)
        rvb.startArray(nCols.toInt)
        var colsRemaining = nCols.toInt
        pfs.foreach { pf =>
          val colsAdded = pf.addRow(rvb, colsRemaining)
          colsRemaining -= colsAdded
        }
        assert(colsRemaining == 0)
        rvb.endArray()
        if (createRowUID) rvb.addLong(row)
        rvb.endStruct()
        rvb.end()
      }
    }
  }

  override def getPartitions: Array[Partition] =
    Array.tabulate(partitionRanges.length)(pi => new Partition { val index: Int = pi })
}

class BlockMatrixCachedPartFile(
  private[this] val startRow: Int,
  _cacheCapacity: Int,
  private[this] val fs: FS,
  path: String,
  pFile: String,
) {
  private[this] val cacheCapacity = math.min(_cacheCapacity, BlockMatrix.bufferSpecBlockSize)
  private[this] val cache = new Array[Double](cacheCapacity)
  private[this] var cacheIndex = cacheCapacity
  private[this] var cacheEnd = cacheCapacity
  private[this] var fileIndex = 0
  private[this] val filename = path + "/parts/" + pFile
  private[this] var rows = -1
  private[this] var cols = -1

  private[this] var row = startRow

  using(fs.open(filename)) { is =>
    val in = BlockMatrix.bufferSpec.buildInputBuffer(is)
    rows = in.readInt()
    assert(rows > 0)
    cols = in.readInt()
    assert(cols > 0)
    assert(cols <= cacheCapacity)
    val isTranspose = in.readBoolean()
    assert(isTranspose, "BlockMatrix must be saved in row-major format")
    in.skipBytes(startRow * cols * 8)
    val doublesToRead = math.min(cacheCapacity, (rows - startRow) * cols)
    in.readDoubles(cache, 0, doublesToRead)
    cacheIndex = 0
    cacheEnd = doublesToRead
    fileIndex = startRow * cols + doublesToRead
    log.info(s"fileIndex 1 $fileIndex")
  }

  private[this] def fillCache(): Unit = {
    System.arraycopy(cache, cacheIndex, cache, 0, cacheEnd - cacheIndex)
    val startWritingAt = cacheEnd - cacheIndex
    cacheIndex = 0
    using(fs.open(filename)) { is =>
      val in = BlockMatrix.bufferSpec.buildInputBuffer(is)

      assert(rows == in.readInt())
      assert(cols == in.readInt())
      val isTranspose = in.readBoolean()
      assert(isTranspose, "BlockMatrix must be saved in row-major format")

      in.skipBytes(8 * fileIndex)
      val doublesToRead = math.min(
        cacheCapacity - startWritingAt,
        rows * cols - fileIndex,
      )
      in.readDoubles(cache, startWritingAt, doublesToRead)
      cacheEnd = doublesToRead + startWritingAt
      var i = 0
      fileIndex += doublesToRead
      assert(doublesToRead > 0)
    }
  }

  def addRow(rvb: RegionValueBuilder, colsRemaining: Int): Int = {
    assert(cols <= colsRemaining)
    if (cacheIndex + cols > cacheEnd) {
      fillCache()
    }

    var i = cacheIndex
    val endOfRow = cacheIndex + cols
    while (i < endOfRow) {
      rvb.addDouble(cache(i))
      i += 1
    }

    row += 1
    cacheIndex += cols
    return cols
  }
}
