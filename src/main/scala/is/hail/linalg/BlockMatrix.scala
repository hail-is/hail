package is.hail.linalg

import java.io._

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum => breezeSum, _}
import breeze.numerics.{pow => breezePow, sqrt => breezeSqrt, log => breezeLog, abs => breezeAbs}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import is.hail._
import is.hail.annotations._
import is.hail.table.Table
import is.hail.expr.types._
import is.hail.io._
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.commons.lang3.StringUtils
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.executor.InputMetrics
import org.apache.spark._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s._

object BlockMatrix {
  type M = BlockMatrix
  val defaultBlockSize: Int = 4096 // 32 * 1024 bytes
  val bufferSpec: BufferSpec =
    new BlockingBufferSpec(32 * 1024,
      new LZ4BlockBufferSpec(32 * 1024,
        new StreamBlockBufferSpec))
  
  def apply(sc: SparkContext, gp: GridPartitioner, piBlock: (GridPartitioner, Int) => ((Int, Int), BDM[Double])): BlockMatrix =
    new BlockMatrix(
      new RDD[((Int, Int), BDM[Double])](sc, Nil) {
        override val partitioner = Some(gp)
  
        protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
          new Partition { def index: Int = pi } )
  
        def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
          Iterator.single(piBlock(gp, split.index))
      }, gp.blockSize, gp.nRows, gp.nCols)
  
  def fromBreezeMatrix(sc: SparkContext, lm: BDM[Double]): M =
    fromBreezeMatrix(sc, lm, defaultBlockSize)

  def fromBreezeMatrix(sc: SparkContext, lm: BDM[Double], blockSize: Int): M = {
    val gp = GridPartitioner(blockSize, lm.rows, lm.cols)
    
    val localBlocksBc = Array.tabulate(gp.numPartitions) { pi =>
      val (i, j) = gp.blockCoordinates(pi)
      val (blockNRows, blockNCols) = gp.blockDims(pi)
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      sc.broadcast(lm(iOffset until iOffset + blockNRows, jOffset until jOffset + blockNCols).copy)
    }
    
    BlockMatrix(sc, gp, (gp, pi) => (gp.blockCoordinates(pi), localBlocksBc(pi).value))
  }

  def fromIRM(irm: IndexedRowMatrix): M =
    fromIRM(irm, defaultBlockSize)

  def fromIRM(irm: IndexedRowMatrix, blockSize: Int): M =
    irm.toHailBlockMatrix(blockSize)

  def fill(hc: HailContext, nRows: Long, nCols: Long, value: Double, blockSize: Int = defaultBlockSize): BlockMatrix =
    BlockMatrix(hc.sc, GridPartitioner(blockSize, nRows, nCols), (gp, pi) => {
      val (i, j) = gp.blockCoordinates(pi)
      ((i, j), BDM.fill[Double](gp.blockRowNRows(i), gp.blockColNCols(j))(value))
    })
  
  // uniform or Gaussian
  def random(hc: HailContext, nRows: Int, nCols: Int, blockSize: Int = defaultBlockSize,
    seed: Int = 0, gaussian: Boolean): M =
    BlockMatrix(hc.sc, GridPartitioner(blockSize, nRows, nCols), (gp, pi) => {
      val (i, j) = gp.blockCoordinates(pi)
      val blockSeed = seed + 15485863 * pi // millionth prime

      val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(blockSeed)))
      val rand = if (gaussian) randBasis.gaussian else randBasis.uniform
      
      ((i, j), BDM.rand[Double](gp.blockRowNRows(i), gp.blockColNCols(j), rand))
    } )

  def map2(f: (Double, Double) => Double)(l: M, r: M): M =
    l.map2(r, f)

  def map4(f: (Double, Double, Double, Double) => Double)(a: M, b: M, c: M, d: M): M =
    a.map4(b, c, d, f)

  val metadataRelativePath = "/metadata.json"

  def checkWriteSuccess(hc: HailContext, uri: String) {
    if (!hc.hadoopConf.exists(uri + "/_SUCCESS"))
      fatal(s"Error reading block matrix. Earlier write failed: no success indicator found at uri $uri")    
  }
  
  def readMetadata(hc: HailContext, uri: String): BlockMatrixMetadata = {
    hc.hadoopConf.readTextFile(uri + metadataRelativePath) { isr =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.read[BlockMatrixMetadata](isr)
    }
  }

  def read(hc: HailContext, uri: String): M = {
    checkWriteSuccess(hc, uri)

    val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) = readMetadata(hc, uri)

    val gp = GridPartitioner(blockSize, nRows, nCols, maybeFiltered)

    def readBlock(pi: Int, is: InputStream, metrics: InputMetrics): Iterator[((Int, Int), BDM[Double])] = {
      val block = RichDenseMatrixDouble.read(is, bufferSpec)
      is.close()

      Iterator.single(gp.partCoordinates(pi), block)
    }

    val blocks = hc.readPartitions(uri, partFiles, readBlock, Some(gp))

    new BlockMatrix(blocks, blockSize, nRows, nCols)
  }

  private[linalg] def assertCompatibleLocalMatrix(lm: BDM[Double]) {
    assert(lm.isCompact)
  }

  private[linalg] def block(bm: BlockMatrix, parts: Array[Partition], gp: GridPartitioner, context: TaskContext,
    i: Int, j: Int): Option[BDM[Double]] = {
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

  // rectangles checked in Python
  def exportRectangles(
    hc: HailContext,
    input: String,
    output: String,
    flattenedRectangles: Array[Long],
    delimiter: String,
    nPartitions: Int): Unit = {
    require(flattenedRectangles.length % 4 == 0)

    checkWriteSuccess(hc, input)

    val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) = BlockMatrix.readMetadata(hc, input)
    val gp = GridPartitioner(blockSize, nRows, nCols, maybeFiltered)

    val rectangles = flattenedRectangles.grouped(4).toArray    
    val dRect = digitsNeeded(rectangles.length)

    val sHadoopBc = hc.sc.broadcast(new SerializableHadoopConfiguration(hc.hadoopConf))
    val partFilesBc = hc.sc.broadcast(partFiles)

    val rdd = hc.sc.parallelize(rectangles.zipWithIndex, numSlices = nPartitions).map { case (r, index) =>
      val partFiles = partFilesBc.value
      val data = new Array[Double](blockSize)
      val sb = new StringBuilder(blockSize << 2)
      val paddedIndex = StringUtils.leftPad(index.toString, dRect, "0")
      val outputFile = output + "/rect-" + paddedIndex + "_" + r.mkString("-")

      val osw = new OutputStreamWriter(sHadoopBc.value.value.unsafeWriter(outputFile))
      try {
        val startRow = r(0)
        val stopRow = r(1)
        val startCol = r(2)
        val stopCol = r(3)

        val nonEmpty = startRow < stopRow && startCol < stopCol

        if (nonEmpty) {
          val startRowOffset = gp.indexBlockOffset(startRow)

          val startBlockCol = gp.indexBlockIndex(startCol)
          val startColOffset = gp.indexBlockOffset(startCol)

          val stopBlockCol = gp.indexBlockIndex(stopCol - 1) + 1
          val stopColOffset = gp.indexBlockOffset(stopCol - 1) + 1

          val startColByteOffset = startColOffset << 3
          val stopColByteDeficit = (gp.blockColNCols(stopBlockCol - 1) - stopColOffset) << 3

          val inPerBlockCol = new Array[InputBuffer](stopBlockCol - startBlockCol)
          try {
            var i = startRow
            while (i < stopRow) {
              if (i == startRow || gp.indexBlockOffset(i) == 0) {
                val blockRow = gp.indexBlockIndex(i)
                val nRowsInBlock = gp.blockRowNRows(blockRow)

                var blockCol = startBlockCol
                while (blockCol < stopBlockCol) {
                  val pi = gp.coordinatesPart(blockRow, blockCol)
                  if (pi < 0)
                    fatal(s"block ($blockRow, $blockCol) missing for rectangle $index " +
                      s"with bounds ${ r.mkString("[", ", ", "]") }")

                  val is = sHadoopBc.value.value.unsafeReader(input + "/parts/" + partFiles(pi))
                  val in = BlockMatrix.bufferSpec.buildInputBuffer(is)

                  val nColsInBlock = gp.blockColNCols(blockCol)

                  assert(in.readInt() == nRowsInBlock)
                  assert(in.readInt() == nColsInBlock)
                  val isTranspose = in.readBoolean()
                  if (!isTranspose)
                    fatal("BlockMatrix must be stored row major on disk in order to export rectangular regions.")

                  if (i == startRow) {
                    val skip = startRowOffset * (nColsInBlock << 3)
                    in.skipBytes(skip)
                  }

                  inPerBlockCol(blockCol - startBlockCol) = in

                  blockCol += 1
                }
              }

              inPerBlockCol.head.skipBytes(startColByteOffset)

              var blockCol = startBlockCol
              while (blockCol < stopBlockCol) {
                val startColOffsetInBlock =
                  if (blockCol > startBlockCol)
                    0
                  else
                    startColOffset

                val stopColOffsetInBlock =
                  if (blockCol < stopBlockCol - 1)
                    blockSize
                  else
                    stopColOffset

                val n = stopColOffsetInBlock - startColOffsetInBlock

                inPerBlockCol(blockCol - startBlockCol).readDoubles(data, 0, n)

                sb.clear()
                var k = 0
                while (k < n - 1) {
                  sb.append(data(k))
                  sb.append(delimiter)
                  k += 1
                }
                sb.append(data(n - 1))
                if (blockCol < stopBlockCol)
                  sb.append(delimiter)
                else
                  sb.append("\n")

                osw.write(sb.result())

                blockCol += 1
              }
              i += 1

              inPerBlockCol.last.skipBytes(stopColByteDeficit)

              if (i % blockSize == 0 && i < stopRow)
                inPerBlockCol.foreach(_.close())
            }
          } finally {
            inPerBlockCol.foreach(in => if (in != null) in.close())
          }
        }
      } finally {
        osw.close()
      }
      
      1
    }

    val nFiles = rdd.reduce(_ + _)
    assert(nFiles == rectangles.length)
    
    info(s"wrote $nFiles rectangular regions to files")
  }

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
}

// must be top-level for Jackson to serialize correctly
case class BlockMatrixMetadata(blockSize: Int, nRows: Long, nCols: Long, maybeFiltered: Option[Array[Int]], partFiles: Array[String])

class BlockMatrix(val blocks: RDD[((Int, Int), BDM[Double])],
  val blockSize: Int,
  val nRows: Long,
  val nCols: Long) extends Serializable {

  import BlockMatrix._

  private[linalg] val st: String = Thread.currentThread().getStackTrace.mkString("\n")

  require(blocks.partitioner.isDefined)
  require(blocks.partitioner.get.isInstanceOf[GridPartitioner])

  val gp: GridPartitioner = blocks.partitioner.get.asInstanceOf[GridPartitioner]
  
  require(gp.blockSize == blockSize && gp.nRows == nRows && gp.nCols == nCols)
  
  val isSparse: Boolean = gp.maybeBlocks.isDefined
  
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
  def realizeBlocks(maybeBlocksToRealize: Option[Array[Int]]): BlockMatrix = {    
    val realizeGP = gp.copy(maybeBlocks =
      if (maybeBlocksToRealize.exists(_.length == gp.maxNBlocks)) None else maybeBlocksToRealize)

    val newGP = gp.union(realizeGP)
    
    if (newGP.numPartitions == gp.numPartitions)
      this
    else {
      def newPIPartition(pi: Int): Iterator[((Int, Int), BDM[Double])] = {
        val bi = newGP.partBlock(pi)
        val lm = (BDM.zeros[Double] _).tupled(newGP.blockDims(bi))
        Iterator.single((newGP.blockCoordinates(bi), lm))
      }
      val oldToNewPI = gp.maybeBlocks.get.map(newGP.blockPart)
      val newBlocks = blocks.supersetPartitions(oldToNewPI, newGP.numPartitions, newPIPartition, Some(newGP))

      new BlockMatrix(newBlocks, blockSize, nRows, nCols)
    }
  }

  def filterBlocks(blocksToKeep: Array[Int]): BlockMatrix =
    if (blocksToKeep.length == gp.maxNBlocks)
      this
    else
      subsetBlocks(gp.intersect(gp.copy(maybeBlocks = Some(blocksToKeep))))
    
  // assumes subsetGP blocks are subset of gp blocks, as with subsetGP = gp.intersect(gp2)
  def subsetBlocks(subsetGP: GridPartitioner): BlockMatrix = {
    if (subsetGP.numPartitions == gp.numPartitions)
      this
    else {
      assert(subsetGP.maybeBlocks.isDefined)
      new BlockMatrix(blocks.subsetPartitions(subsetGP.maybeBlocks.get.map(gp.blockPart), Some(subsetGP)),
        blockSize, nRows, nCols)
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
    val zeroedBlocks = blocks.mapPartitions( { it =>
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
    }, preservesPartitioning = true)
    
    new BlockMatrix(zeroedBlocks, blockSize, nRows, nCols)
  }
  
  // for row i, filter to indices [starts[i], stops[i]) by dropping non-overlapping blocks
  // if not blocksOnly, also zero out elements outside ranges in overlapping blocks
  // checked in Python: start >= 0 && start <= stop && stop <= nCols
  def filterRowIntervals(starts: Array[Long], stops: Array[Long], blocksOnly: Boolean): BlockMatrix = {
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
    val sc = blocks.sparkContext
    val startBlockIndexBc = sc.broadcast(starts.map(gp.indexBlockIndex))
    val stopBlockIndexBc = sc.broadcast(stops.map(stop => (stop / blockSize).toInt))
    val startBlockOffsetBc = sc.broadcast(starts.map(gp.indexBlockOffset))
    val stopBlockOffsetsBc = sc.broadcast(stops.map(gp.indexBlockOffset))

    val zeroedBlocks = blocks.mapPartitions( { it =>
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
    }, preservesPartitioning = true)
    
    new BlockMatrix(zeroedBlocks, blockSize, nRows, nCols)
  }
  
  def filterRectangles(flattenedRectangles: Array[Long]): BlockMatrix = {
    require(flattenedRectangles.length % 4 == 0)
    val rectangles = flattenedRectangles.grouped(4).toArray
    
    filterBlocks(gp.rectanglesBlocks(rectangles))
  }
  
  // element-wise ops
  def unary_+(): M = this
  
  def unary_-(): M = blockMap(-_,
    "negation",
    reqDense = false)

  def add(that: M): M =
    if (sameBlocks(that)) {
      blockMap2(that, _ + _,
        "addition",
        reqDense = false)
    } else {
      val addBlocks = new BlockMatrixUnionOpRDD(this, that,
        _ match {
          case (Some(a), Some(b)) => a + b
          case (Some(a), None) => a
          case (None, Some(b)) => b
          case (None, None) => fatal("not possible for union")
        }
      )
      new BlockMatrix(addBlocks, blockSize, nRows, nCols)
    }
  
  def sub(that: M): M =
    if (sameBlocks(that)) {
      blockMap2(that, _ - _,
        "subtraction",
        reqDense = false)
    } else {
      val subBlocks = new BlockMatrixUnionOpRDD(this, that,
        _ match {
          case (Some(a), Some(b)) => a - b
          case (Some(a), None) => a
          case (None, Some(b)) => -b
          case (None, None) => fatal("not possible for union")
        }
      )
      new BlockMatrix(subBlocks, blockSize, nRows, nCols)
    }
  
  def mul(that: M): M = {
    val newGP = gp.intersect(that.gp)
    subsetBlocks(newGP).blockMap2(
      that.subsetBlocks(newGP), _ *:* _,
      "element-wise multiplication",
      reqDense = false)
  }
  
  def div(that: M): M = blockMap2(that, _ /:/ _,
    "element-wise division")
  
  // row broadcast
  def rowVectorAdd(a: Array[Double]): M = densify().rowVectorOp((lm, lv) => lm(*, ::) + lv,
    "broadcasted addition of row-vector")(a)
  
  def rowVectorSub(a: Array[Double]): M = densify().rowVectorOp((lm, lv) => lm(*, ::) - lv,
    "broadcasted subtraction of row-vector")(a)
  
  def rowVectorMul(a: Array[Double]): M = rowVectorOp((lm, lv) => lm(*, ::) *:* lv,
    "broadcasted multiplication by row-vector containing nan, or infinity",
    reqDense = a.exists(i => i.isNaN | i.isInfinity))(a)
  
  def rowVectorDiv(a: Array[Double]): M = rowVectorOp((lm, lv) => lm(*, ::) /:/ lv, 
    "broadcasted division by row-vector containing zero, nan, or infinity",
    reqDense = a.exists(i => i == 0.0 | i.isNaN | i.isInfinity))(a)

  def reverseRowVectorSub(a: Array[Double]): M = densify().rowVectorOp((lm, lv) => lm(*, ::).map(lv - _),
    "broadcasted row-vector minus block matrix")(a)
 
  def reverseRowVectorDiv(a: Array[Double]): M = rowVectorOp((lm, lv) => lm(*, ::).map(lv /:/ _),
    "broadcasted row-vector divided by block matrix")(a)
  
  // column broadcast
  def colVectorAdd(a: Array[Double]): M = densify().colVectorOp((lm, lv) => lm(::, *) + lv,
    "broadcasted addition of column-vector")(a)
  
  def colVectorSub(a: Array[Double]): M = densify().colVectorOp((lm, lv) => lm(::, *) - lv,
    "broadcasted subtraction of column-vector")(a)
  
  def colVectorMul(a: Array[Double]): M = colVectorOp((lm, lv) => lm(::, *) *:* lv,
    "broadcasted multiplication column-vector containing nan or infinity",
    reqDense = a.exists(i => i.isNaN | i.isInfinity))(a)
  
  def colVectorDiv(a: Array[Double]): M = colVectorOp((lm, lv) => lm(::, *) /:/ lv,
    "broadcasted division by column-vector containing zero, nan, or infinity",
    reqDense = a.exists(i => i == 0.0 | i.isNaN | i.isInfinity))(a)

  def reverseColVectorSub(a: Array[Double]): M = densify().colVectorOp((lm, lv) => lm(::, *).map(lv - _),
    "broadcasted column-vector minus block matrix")(a)

  def reverseColVectorDiv(a: Array[Double]): M = colVectorOp((lm, lv) => lm(::, *).map(lv /:/ _),
    "broadcasted column-vector divided by block matrix")(a)

  // scalar
  def scalarAdd(i: Double): M = densify().blockMap(_ + i,
    "scalar addition")
  
  def scalarSub(i: Double): M = densify().blockMap(_ - i,
    "scalar subtraction")
  
  def scalarMul(i: Double): M = blockMap(_ *:* i,
      s"multiplication by scalar $i",
      reqDense = i.isNaN | i.isInfinity)
  
  def scalarDiv(i: Double): M = blockMap(_ /:/ i,
      s"division by scalar $i",
      reqDense = i == 0.0 | i.isNaN | i.isInfinity)
  
  def reverseScalarSub(i: Double): M = densify().blockMap(i - _,
    s"scalar minus block matrix")
  
  def reverseScalarDiv(i: Double): M = blockMap(i /:/ _,
    s"scalar divided by block matrix")

  // other element-wise ops
  def sqrt(): M = blockMap(breezeSqrt(_),
    "sqrt",
    reqDense = false)

  def pow(exponent: Double): M = blockMap(breezePow(_, exponent),
    s"exponentiation by negative power $exponent",
    reqDense = exponent < 0)
  
  def log(): M = blockMap(breezeLog(_),
    "natural logarithm")

  def abs(): M = blockMap(breezeAbs(_),
    "absolute value",
    reqDense = false)
  
  // matrix ops
  def dot(that: M): M = new BlockMatrix(new BlockMatrixMultiplyRDD(this, that), blockSize, nRows, that.nCols)

  def dot(lm: BDM[Double]): M = {
    require(nCols == lm.rows,
      s"incompatible matrix dimensions: ${ nRows } x ${ nCols } and ${ lm.rows } x ${ lm.cols }")
    dot(BlockMatrix.fromBreezeMatrix(blocks.sparkContext, lm, blockSize))
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

  def write(uri: String, forceRowMajor: Boolean = false) {
    val hadoop = blocks.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    def writeBlock(it: Iterator[((Int, Int), BDM[Double])], os: OutputStream): Int = {
      assert(it.hasNext)
      val (_, lm) = it.next()
      assert(!it.hasNext)

      lm.write(os, forceRowMajor, bufferSpec)
      os.close()

      1
    }

    val (partFiles, _) = blocks.writePartitions(uri, writeBlock)

    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, nCols, gp.maybeBlocks, partFiles),
        os)
    }

    hadoop.writeTextFile(uri + "/_SUCCESS")(out => ())
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

  def toBreezeMatrix(): BDM[Double] = {
    require(nRows <= Int.MaxValue, "The number of rows of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently nRows: $nRows")
    require(nCols <= Int.MaxValue, "The number of columns of this matrix should be less than or equal to " +
      s"Int.MaxValue. Currently nCols: $nCols")
    require(nRows * nCols <= Int.MaxValue, "The length of the values array must be " +
      s"less than or equal to Int.MaxValue. Currently nRows * nCols: ${ nRows * nCols }")
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

  private def requireZippable(that: M, name: String = "operation") {
    require(nRows == that.nRows,
      s"$name requires same number of rows, but actually: ${ nRows }x${ nCols }, ${ that.nRows }x${ that.nCols }")
    require(nCols == that.nCols,
      s"$name requires same number of cols, but actually: ${ nRows }x${ nCols }, ${ that.nRows }x${ that.nCols }")
    require(blockSize == that.blockSize,
      s"$name requires same block size, but actually: $blockSize and ${ that.blockSize }")
    if (!sameBlocks(that))
      fatal(s"$name requires block matrices to have the same set of blocks present")
  }
  
  private def sameBlocks(that: M): Boolean = {
    (gp.maybeBlocks, that.gp.maybeBlocks) match {
      case (Some(bis), Some(bis2)) => bis sameElements bis2
      case (None, None) => true
      case _ => false
    }
  }
  
  def blockMap(op: BDM[Double] => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense)
      requireDense(name)
    new BlockMatrix(blocks.mapValues(op), blockSize, nRows, nCols)
  }
  
  def blockMapWithIndex(op: ((Int, Int), BDM[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense)
      requireDense(name)
    new BlockMatrix(blocks.mapValuesWithKey(op), blockSize, nRows, nCols)
  }

  def blockMap2(that: M,
    op: (BDM[Double], BDM[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks = blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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

  def map(op: Double => Double,
    name: String = "operation",
    reqDense: Boolean = true): M = {
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

  def map2(that: M,
    op: (Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks = blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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

  def map4(bm2: M, bm3: M, bm4: M,
    op: (Double, Double, Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense) {
      requireDense(name)
      bm2.requireDense(name)
      bm3.requireDense(name)
      bm4.requireDense(name)
    }
    requireZippable(bm2)
    requireZippable(bm3)
    requireZippable(bm4)
    val newBlocks = blocks.zipPartitions(bm2.blocks, bm3.blocks, bm4.blocks, preservesPartitioning = true) { (it1, it2, it3, it4) =>
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def mapWithIndex(op: (Long, Long, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true): M = {
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

  def map2WithIndex(that: M,
    op: (Long, Long, Double, Double) => Double,
    name: String = "operation",
    reqDense: Boolean = true): M = {
    if (reqDense) {
      requireDense(name)
      that.requireDense(name)
    }
    requireZippable(that)
    val newBlocks = blocks.zipPartitions(that.blocks, preservesPartitioning = true) { (thisIter, thatIter) =>
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
    new BlockMatrix(newBlocks, blockSize, nRows, nCols)
  }

  def colVectorOp(op: (BDM[Double], BDV[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true): Array[Double] => M = {
    a => val v = BDV(a)
      require(v.length == nRows, s"vector length must equal nRows: ${ v.length }, $nRows")
      val vBc = blocks.sparkContext.broadcast(v)
      blockMapWithIndex( { case ((i, _), lm) =>
        val lv = gp.vectorOnBlockRow(vBc.value, i)
        op(lm, lv)
      }, name, reqDense = reqDense)
  }

  def rowVectorOp(op: (BDM[Double], BDV[Double]) => BDM[Double],
    name: String = "operation",
    reqDense: Boolean = true): Array[Double] => M = {
    a => val v = BDV(a)
      require(v.length == nCols, s"vector length must equal nCols: ${ v.length }, $nCols")
      val vBc = blocks.sparkContext.broadcast(v)
      blockMapWithIndex( { case ((_, j), lm) =>
        val lv = gp.vectorOnBlockCol(vBc.value, j)
        op(lm, lv)
      }, name, reqDense = reqDense)
  }

  def reduce(blockOp: BDM[Double] => Double, scalarOp: (Double, Double) => Double): Double =
      blocks
        .map { case ((i, j), lm) => blockOp(lm) }
        .fold(0.0)(scalarOp)

  def rowReduce(blockOp: BDM[Double] => BDV[Double], vectorOp: (BDV[Double], BDV[Double]) => BDV[Double]): BlockMatrix =
    new BlockMatrix(
      blocks
        .map { case ((i, j), lm) => ((0, j), blockOp(lm)) }
        .reduceByKey(GridPartitioner(blockSize, 1, nCols, gp.maybeBlockCols()), vectorOp)
        .mapValues(v => new BDM[Double](1, v.length, v.data)),
      blockSize, 1, nCols)
  
  def colReduce(blockOp: BDM[Double] => BDV[Double], vectorOp: (BDV[Double], BDV[Double]) => BDV[Double]): BlockMatrix =    
    new BlockMatrix(
      blocks
        .map { case ((i, j), lm) => ((i, 0), blockOp(lm)) }
        .reduceByKey(GridPartitioner(blockSize, nRows, 1, gp.maybeBlockRows()), vectorOp)
        .mapValues(v => new BDM[Double](v.length, 1, v.data)),
      blockSize, nRows, 1)
  
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

    new IndexedRowMatrix(blocks.flatMap { case ((i, j), lm) =>
      val iOffset = i * blockSize
      val jOffset = j * blockSize

      for (k <- 0 until lm.rows)
        yield (k + iOffset, (jOffset, lm(k, ::).inner.toArray))
    }.aggregateByKey(new Array[Double](nColsInt))(seqOp, combOp)
      .map { case (i, a) => IndexedRow(i, BDV(a)) },
      nRows, nColsInt)
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
  
  def filterRows(keep: Array[Long]): BlockMatrix = {
    transpose().filterCols(keep).transpose()
  }

  def filterCols(keep: Array[Long]): BlockMatrix = {
    new BlockMatrix(new BlockMatrixFilterColsRDD(densify(), keep), blockSize, nRows, keep.length)
  }

  def filter(keepRows: Array[Long], keepCols: Array[Long]): BlockMatrix = {
    new BlockMatrix(new BlockMatrixFilterRDD(densify(), keepRows, keepCols),
      blockSize, keepRows.length, keepCols.length)
  }

  def entriesTable(hc: HailContext): Table = {
    val rvRowType = TStruct("i" -> TInt64Optional, "j" -> TInt64Optional, "entry" -> TFloat64Optional)
    
    val entriesRDD = ContextRDD.weaken[RVDContext](blocks).cflatMap { case (ctx, ((blockRow, blockCol), block)) =>
      val rowOffset = blockRow * blockSize.toLong
      val colOffset = blockCol * blockSize.toLong

      val region = ctx.region
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      block.activeIterator
        .map { case ((i, j), entry) =>
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

    new Table(hc, entriesRDD, rvRowType, Some(Array("i", "j")))
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

    computeAllBlockColRanges(keep, gp.transpose._1, newGP.transpose._1)
  }
}

// checked in Python: keepRows and keepCols non-empty, increasing, valid range
private class BlockMatrixFilterRDD(bm: BlockMatrix, keepRows: Array[Long], keepCols: Array[Long])
  extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {
  
  private val gp = bm.gp
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
    new NarrowDependency(bm.blocks) {
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
        val (_, block) = bm.blocks.iterator(bm.blocks.partitions(parentPI), context).next()

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

// checked in Python: keep non-empty, increasing, valid range
private class BlockMatrixFilterColsRDD(bm: BlockMatrix, keep: Array[Long])
  extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {

  private val gp = bm.gp
  private val blockSize = gp.blockSize
  private val newGP = GridPartitioner(blockSize, gp.nRows, keep.length)

  private val allBlockColRanges: Array[Array[(Int, Array[Int], Array[Int])]] =
    BlockMatrixFilterRDD.computeAllBlockColRanges(keep, gp, newGP)

  protected def getPartitions: Array[Partition] =
    Array.tabulate(newGP.numPartitions) { pi =>
      BlockMatrixFilterColsRDDPartition(pi, allBlockColRanges(newGP.blockBlockCol(pi)))
    }

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
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
        val (_, block) = bm.blocks.iterator(bm.blocks.partitions(parentPI), context).next()

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

private class BlockMatrixTransposeRDD(bm: BlockMatrix)
  extends RDD[((Int, Int), BDM[Double])](bm.blocks.sparkContext, Nil) {

  private val (newGP, inverseTransposePI) = bm.gp.transpose

  override def getDependencies: Seq[Dependency[_]] = Array[Dependency[_]](
    new NarrowDependency(bm.blocks) {
      def getParents(partitionId: Int): Seq[Int] = Array(inverseTransposePI(partitionId))
    })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] =
    bm.blocks.iterator(split.asInstanceOf[BlockMatrixTransposeRDDPartition].prevPartition, context)
      .map { case ((j, i), lm) => ((i, j), lm.t) }

  protected def getPartitions: Array[Partition] = {
    Array.tabulate(newGP.numPartitions) { pi =>
      BlockMatrixTransposeRDDPartition(pi, bm.blocks.partitions(inverseTransposePI(pi)))
    }
  }

  @transient override val partitioner: Option[Partitioner] = Some(newGP)
}

private class BlockMatrixUnionOpRDD(
  l: BlockMatrix,
  r: BlockMatrix,
  op: ((Option[BDM[Double]], Option[BDM[Double]])) => BDM[Double])
  extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {

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
        def getParents(partitionId: Int): Seq[Int] = Array(lGP.blockPart(gp.partBlock(partitionId))).filter(_ >= 0)
      },
      new NarrowDependency(r.blocks) {
        def getParents(partitionId: Int): Seq[Int] = Array(rGP.blockPart(gp.partBlock(partitionId))).filter(_ >= 0)
      })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (i, j) = gp.partCoordinates(split.index)
    val lm = op(block(l, lParts, lGP, context, i, j), block(r, rParts, rGP, context, i, j))
    
    Iterator.single(((i, j), lm))
  }

  protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
    new Partition { def index: Int = pi } )

  @transient override val partitioner: Option[Partitioner] = Some(gp)
}

private class BlockMatrixMultiplyRDD(l: BlockMatrix, r: BlockMatrix)
  extends RDD[((Int, Int), BDM[Double])](l.blocks.sparkContext, Nil) {

  import BlockMatrix.block

  require(l.nCols == r.nRows,
    s"inner dimensions must match, but given: ${ l.nRows }x${ l.nCols }, ${ r.nRows }x${ r.nCols }")
  require(l.blockSize == r.blockSize,
    s"blocks must be same size, but actually were ${ l.blockSize }x${ l.blockSize } and ${ r.blockSize }x${ r.blockSize }")

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
      })

  def compute(split: Partition, context: TaskContext): Iterator[((Int, Int), BDM[Double])] = {
    val (i, j) = gp.blockCoordinates(split.index)
    val (blockNRows, blockNCols) = gp.blockDims(split.index)
    val product = BDM.zeros[Double](blockNRows, blockNCols)   
    var k = 0
    while (k < nProducts) {
      block(l, lParts, lGP, context, i, k).foreach(left =>
        block(r, rParts, rGP, context, k, j).foreach(right =>
          product :+= left * right))
      k += 1
    }

    Iterator.single(((i, j), product))
  }

  protected def getPartitions: Array[Partition] = Array.tabulate(gp.numPartitions)(pi =>
    new Partition { def index: Int = pi } )

  @transient override val partitioner: Option[Partitioner] = Some(gp)
}

// On compute, WriteBlocksRDDPartition writes the block row with index `index`
// [`start`, `end`] is the range of indices of parent partitions overlapping this block row
// `skip` is the index in the start partition corresponding to the first row of this block row
case class WriteBlocksRDDPartition(index: Int, start: Int, skip: Int, end: Int) extends Partition {
  def range: Range = start to end
}

class WriteBlocksRDD(path: String,
  crdd: ContextRDD[RVDContext, RegionValue],
  sc: SparkContext,
  matrixType: MatrixType,
  parentPartStarts: Array[Long],
  entryField: String,
  gp: GridPartitioner) extends RDD[(Int, String)](sc, Nil) {

  require(gp.nRows == parentPartStarts.last)

  private val parentParts = crdd.partitions
  private val blockSize = gp.blockSize

  private val d = digitsNeeded(gp.numPartitions)
  private val sHadoopBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

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

  def compute(split: Partition, context: TaskContext): Iterator[(Int, String)] = {
    val blockRow = split.index
    val nRowsInBlock = gp.blockRowNRows(blockRow)
    val ctx = TaskContext.get

    val (blockPartFiles, outPerBlockCol) = Array.tabulate(gp.nBlockCols) { blockCol =>
      val nColsInBlock = gp.blockColNCols(blockCol)

      val i = gp.coordinatesBlock(blockRow, blockCol)
      val f = partFile(d, i, ctx)
      val filename = path + "/parts/" + f

      val os = sHadoopBc.value.value.unsafeWriter(filename)
      val out = BlockMatrix.bufferSpec.buildOutputBuffer(os)

      out.writeInt(nRowsInBlock)
      out.writeInt(nColsInBlock)
      out.writeBoolean(true) // transposed, stored row major

      ((i, f), out)
    }
      .unzip

    val rvRowType = matrixType.rvRowType
    val entryArrayType = matrixType.entryArrayType
    val entryType = matrixType.entryType
    val fieldType = entryType.field(entryField).typ

    assert(fieldType.isOfType(TFloat64()))

    val entryArrayIdx = matrixType.entriesIdx
    val fieldIdx = entryType.fieldIdx(entryField)

    val writeBlocksPart = split.asInstanceOf[WriteBlocksRDDPartition]
    val start = writeBlocksPart.start
    writeBlocksPart.range.foreach { pi =>
      using(crdd.mkc()) { ctx =>
        val it = crdd.iterator(parentParts(pi), context, ctx)

        if (pi == start) {
          var j = 0
          while (j < writeBlocksPart.skip) {
            it.next()
            ctx.region.clear()
            j += 1
          }
        }

        val data = new Array[Double](blockSize)

        var i = 0
        while (it.hasNext && i < nRowsInBlock) {
          val rv = it.next()
          val region = rv.region

          val entryArrayOffset = rvRowType.loadField(rv, entryArrayIdx)

          var blockCol = 0
          var colIdx = 0
          while (blockCol < gp.nBlockCols) {
            val n = gp.blockColNCols(blockCol)
            var j = 0
            while (j < n) {
              if (entryArrayType.isElementDefined(region, entryArrayOffset, colIdx)) {
                val entryOffset = entryArrayType.loadElement(region, entryArrayOffset, colIdx)
                if (entryType.isFieldDefined(region, entryOffset, fieldIdx)) {
                  val fieldOffset = entryType.loadField(region, entryOffset, fieldIdx)
                  data(j) = region.loadDouble(fieldOffset)
                } else {
                  val rowIdx = blockRow * blockSize + i
                  fatal(s"Cannot create BlockMatrix: missing value at row $rowIdx and col $colIdx")
                }
              } else {
                val rowIdx = blockRow * blockSize + i
                fatal(s"Cannot create BlockMatrix: missing entry at row $rowIdx and col $colIdx")
              }
              colIdx += 1
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
    blockPartFiles.iterator
  }
}