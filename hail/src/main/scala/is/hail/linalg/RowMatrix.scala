package is.hail.linalg

import breeze.linalg.DenseMatrix
import is.hail.HailContext
import is.hail.backend.{BroadcastValue, ExecuteContext, HailStateManager}
import is.hail.backend.spark.SparkBackend
import is.hail.types.virtual.{TInt64, TStruct}
import is.hail.io.InputBuffer
import is.hail.io.fs.FS
import is.hail.rvd.RVDPartitioner
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, Partitioner, SparkContext, TaskContext}

object RowMatrix {
  def apply(rows: RDD[(Long, Array[Double])], nCols: Int): RowMatrix =
    new RowMatrix(rows, nCols, None, None)
  
  def apply(rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long): RowMatrix =
    new RowMatrix(rows, nCols, Some(nRows), None)
  
  def apply(rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long, partitionCounts: Array[Long]): RowMatrix =
    new RowMatrix(rows, nCols, Some(nRows), Some(partitionCounts))
  
  def computePartitionCounts(partSize: Long, nRows: Long): Array[Long] = {
    val nParts = ((nRows - 1) / partSize).toInt + 1
    val partitionCounts = Array.fill[Long](nParts)(partSize)
    partitionCounts(nParts - 1) = nRows - partSize * (nParts - 1)
    
    partitionCounts
  }

  def readBlockMatrix(fs: FS, uri: String, maybePartSize: java.lang.Integer): RowMatrix = {
    val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) = BlockMatrix.readMetadata(fs, uri)
    if (nCols >= Int.MaxValue) {
      fatal(s"Number of columns must be less than 2^31, found $nCols")
    }
    val gp = GridPartitioner(blockSize, nRows, nCols, maybeFiltered)
    val partSize: Int = if (maybePartSize != null) maybePartSize else blockSize
    val partitionCounts = computePartitionCounts(partSize, gp.nRows)
    RowMatrix(
      new ReadBlocksAsRowsRDD(fs.broadcast, uri, partFiles, partitionCounts, gp),
      gp.nCols.toInt,
      gp.nRows,
      partitionCounts)
  }
}

class RowMatrix(val rows: RDD[(Long, Array[Double])],
  val nCols: Int,
  private var _nRows: Option[Long],
  private var _partitionCounts: Option[Array[Long]]) extends Serializable {

  require(nCols > 0)
  
  def nRows: Long = _nRows match {
    case Some(nRows) => nRows
    case None =>
      _nRows = Some(partitionCounts().sum)
      nRows
    }
  
  def partitionCounts(): Array[Long] = _partitionCounts match {
    case Some(partitionCounts) => partitionCounts
    case None =>
      _partitionCounts = Some(rows.countPerPartition())
      partitionCounts()
  }
  
  // length nPartitions + 1, first element 0, last element rdd2 count
  def partitionStarts(): Array[Long] = partitionCounts().scanLeft(0L)(_ + _)

  def partitioner(
    partitionKey: Array[String] = Array("idx"),
    kType: TStruct = TStruct("idx" -> TInt64)): RVDPartitioner = {
    
    val partStarts = partitionStarts()

    new RVDPartitioner(HailStateManager(Map.empty), partitionKey, kType,
      Array.tabulate(partStarts.length - 1) { i =>
        val start = partStarts(i)
        val end = partStarts(i + 1)
        Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
      })
  }
  
  def toBreezeMatrix(): DenseMatrix[Double] = {
    require(_nRows.forall(_ <= Int.MaxValue), "The number of rows of this matrix should be less than or equal to " +
        s"Int.MaxValue. Currently numRows: ${ _nRows.get }")
    
    val a = rows.map(_._2).collect()
    val nRowsInt = a.length
    
    require(nRowsInt * nCols.toLong <= Int.MaxValue, "The length of the values array must be " +
      s"less than or equal to Int.MaxValue. Currently rows * cols: ${ nRowsInt * nCols.toLong }")    
    
    new DenseMatrix[Double](nRowsInt, nCols, a.flatten, 0, nCols, isTranspose = true)
  }
  
  def export(ctx: ExecuteContext, path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: String) {
    val localNCols = nCols
    exportDelimitedRowSlices(ctx, path, columnDelimiter, header, addIndex, exportType, _ => 0, _ => localNCols)
  }

  // includes the diagonal
  def exportLowerTriangle(ctx: ExecuteContext, path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: String) {
    val localNCols = nCols
    exportDelimitedRowSlices(ctx, path, columnDelimiter, header, addIndex, exportType, _ => 0, i => math.min(i + 1, localNCols.toLong).toInt)
  }

  def exportStrictLowerTriangle(ctx: ExecuteContext, path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: String) {
    val localNCols = nCols
    exportDelimitedRowSlices(ctx, path, columnDelimiter, header, addIndex, exportType, _ => 0, i => math.min(i, localNCols.toLong).toInt)
  }

  // includes the diagonal
  def exportUpperTriangle(ctx: ExecuteContext, path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: String) {
    val localNCols = nCols
    exportDelimitedRowSlices(ctx, path, columnDelimiter, header, addIndex, exportType, i => math.min(i, localNCols.toLong).toInt, _ => localNCols)
  }  
  
  def exportStrictUpperTriangle(ctx: ExecuteContext, path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: String) {
    val localNCols = nCols
    exportDelimitedRowSlices(ctx, path, columnDelimiter, header, addIndex, exportType, i => math.min(i + 1, localNCols.toLong).toInt, _ => localNCols)
  }
  
  // convert elements in [start, end) of each array to a string, delimited by columnDelimiter, and export
  def exportDelimitedRowSlices(
    ctx: ExecuteContext,
    path: String, 
    columnDelimiter: String,
    header: Option[String],
    addIndex: Boolean,
    exportType: String,
    start: (Long) => Int, 
    end: (Long) => Int) {
    
    genericExport(ctx, path, header, exportType, { (sb, i, v) =>
      if (addIndex) {
        sb.append(i)
        sb.append(columnDelimiter)
      }
      val l = start(i)
      val r = end(i)
      var j = l
      while (j < r) {
        if (j > l)
          sb.append(columnDelimiter)
        sb.append(v(j))
        j += 1
      }
    })
  }

  // uses writeRow to convert each row to a string and writes that string to a file if non-empty
  def genericExport(
    ctx: ExecuteContext,
    path: String, 
    header: Option[String], 
    exportType: String,
    writeRow: (StringBuilder, Long, Array[Double]) => Unit) {
    
    rows.mapPartitions { it =>
      val sb = new StringBuilder()
      it.map { case (index, v) =>
        sb.clear()
        writeRow(sb, index, v)
        sb.result()
      }.filter(_.nonEmpty)
    }.writeTable(ctx, path, header, exportType)
  }
}

// [`start`, `end`) is the row range of partition
case class ReadBlocksAsRowsRDDPartition(index: Int, start: Long, end: Long) extends Partition

class ReadBlocksAsRowsRDD(
  fsBc: BroadcastValue[FS],
  path: String,
  partFiles: IndexedSeq[String],
  partitionCounts: Array[Long],
  gp: GridPartitioner) extends RDD[(Long, Array[Double])](SparkBackend.sparkContext("ReadBlocksAsRowsRDD"), Nil) {
  
  private val partitionStarts = partitionCounts.scanLeft(0L)(_ + _)
  
  if (partitionStarts.last != gp.nRows)
    fatal(s"Error reading BlockMatrix as RowMatrix: expected ${partitionStarts.last} rows in RowMatrix, but found ${gp.nRows} rows in BlockMatrix")

  if (gp.nCols > Int.MaxValue)
      fatal(s"Cannot read BlockMatrix with ${gp.nCols} > Int.MaxValue columns as a RowMatrix")
  
  private val nCols = gp.nCols.toInt
  private val nBlockCols = gp.nBlockCols
  private val blockSize = gp.blockSize

  protected def getPartitions: Array[Partition] = Array.tabulate(partitionStarts.length - 1)(pi =>
    ReadBlocksAsRowsRDDPartition(pi, partitionStarts(pi), partitionStarts(pi + 1)))

  def compute(split: Partition, context: TaskContext): Iterator[(Long, Array[Double])] = {
    val ReadBlocksAsRowsRDDPartition(_, start, end) = split.asInstanceOf[ReadBlocksAsRowsRDDPartition]
    
    var inPerBlockCol: IndexedSeq[(InputBuffer, Int, Int)] = null
    var i = start

    new Iterator[(Long, Array[Double])] {
      def hasNext: Boolean = i < end

      def next(): (Long, Array[Double]) = {
        if (i == start || i % blockSize == 0) {
          val blockRow = (i / blockSize).toInt
          val nRowsInBlock = gp.blockRowNRows(blockRow)
          
          inPerBlockCol = (0 until nBlockCols)
            .flatMap { blockCol =>
              val pi = gp.coordinatesPart(blockRow, blockCol)
              if (pi >= 0) {
                val filename = path + "/parts/" + partFiles(pi)

                val is = fsBc.value.open(filename)
                val in = BlockMatrix.bufferSpec.buildInputBuffer(is)

                val nColsInBlock = gp.blockColNCols(blockCol)

                assert(in.readInt() == nRowsInBlock)
                assert(in.readInt() == nColsInBlock)
                val isTranspose = in.readBoolean()
                if (!isTranspose)
                  fatal("BlockMatrix must be stored row major on disk in order to be read as a RowMatrix")

                if (i == start) {
                  val skip = (start % blockSize).toInt * (nColsInBlock << 3)
                  in.skipBytes(skip)
                }

                Some((in, blockCol, nColsInBlock))
              } else
                None
            }
        }

        val row = new Array[Double](nCols)
        
        inPerBlockCol.foreach { case (in, blockCol, nColsInBlock) =>
          in.readDoubles(row, blockCol * blockSize, nColsInBlock)
        }
        
        val iRow = (i, row)
        
        i += 1
        
        if (i % blockSize == 0 || i == end)
          inPerBlockCol.foreach(_._1.close())
        
        iRow
      }
    }
  }
  
  @transient override val partitioner: Option[Partitioner] = Some(RowPartitioner(partitionStarts))
}
