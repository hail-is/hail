package is.hail.linalg

import breeze.linalg.DenseMatrix
import is.hail.HailContext
import is.hail.expr.types.{TInt64, TStruct}
import is.hail.io.InputBuffer
import is.hail.rvd.OrderedRVDPartitioner
import is.hail.utils._
import org.apache.spark.{Partition, Partitioner, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object RowMatrix {
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int): RowMatrix =
    new RowMatrix(hc, rows, nCols, None, None)
  
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long): RowMatrix =
    new RowMatrix(hc, rows, nCols, Some(nRows), None)
  
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long, partitionCounts: Array[Long]): RowMatrix =
    new RowMatrix(hc, rows, nCols, Some(nRows), Some(partitionCounts))
  
  def computePartitionCounts(partSize: Long, nRows: Long): Array[Long] = {
    val nParts = ((nRows - 1) / partSize).toInt + 1
    val partitionCounts = Array.fill[Long](nParts)(partSize)
    partitionCounts(nParts - 1) = nRows - partSize * (nParts - 1)
    
    partitionCounts
  }

  def readBlockMatrix(hc: HailContext, uri: String, maybePartSize: Option[Int]): RowMatrix = {
    val BlockMatrixMetadata(blockSize, nRows, nCols, maybeFiltered, partFiles) = BlockMatrix.readMetadata(hc, uri)
    if (nCols >= Int.MaxValue) {
      fatal(s"Number of columns must be less than 2^31, found $nCols")
    }
    val gp = GridPartitioner(blockSize, nRows, nCols, maybeFiltered)
    val partSize = maybePartSize.getOrElse(blockSize)
    val partitionCounts = computePartitionCounts(partSize, gp.nRows)
    RowMatrix(hc, 
      new ReadBlocksAsRowsRDD(uri, hc.sc, partFiles, partitionCounts, gp),
      gp.nCols.toInt,
      gp.nRows,
      partitionCounts)
  }
}

class RowMatrix(val hc: HailContext,
  val rows: RDD[(Long, Array[Double])],
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

  def orderedRVDPartitioner(
    partitionKey: Array[String] = Array("idx"),
    kType: TStruct = TStruct("idx" -> TInt64())): OrderedRVDPartitioner = {
    
    val partStarts = partitionStarts()

    new OrderedRVDPartitioner(partitionKey, kType,
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
  
  def export(path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, addIndex, exportType, _ => 0, _ => localNCols)
  }

  // includes the diagonal
  def exportLowerTriangle(path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, addIndex, exportType, _ => 0, i => math.min(i + 1, localNCols.toLong).toInt)
  }

  def exportStrictLowerTriangle(path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, addIndex, exportType, _ => 0, i => math.min(i, localNCols.toLong).toInt)
  }

  // includes the diagonal
  def exportUpperTriangle(path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, addIndex, exportType, i => math.min(i, localNCols.toLong).toInt, _ => localNCols)
  }  
  
  def exportStrictUpperTriangle(path: String, columnDelimiter: String, header: Option[String], addIndex: Boolean, exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, addIndex, exportType, i => math.min(i + 1, localNCols.toLong).toInt, _ => localNCols)
  }
  
  // convert elements in [start, end) of each array to a string, delimited by columnDelimiter, and export
  def exportDelimitedRowSlices(
    path: String, 
    columnDelimiter: String,
    header: Option[String],
    addIndex: Boolean,
    exportType: Int, 
    start: (Long) => Int, 
    end: (Long) => Int) {
    
    genericExport(path, header, exportType, { (sb, i, v) =>
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
    path: String, 
    header: Option[String], 
    exportType: Int, 
    writeRow: (StringBuilder, Long, Array[Double]) => Unit) {
    
    rows.mapPartitions { it =>
      val sb = new StringBuilder()
      it.map { case (index, v) =>
        sb.clear()
        writeRow(sb, index, v)
        sb.result()
      }.filter(_.nonEmpty)
    }.writeTable(path, hc.tmpDir, header, exportType)
  }
}

// [`start`, `end`) is the row range of partition
case class ReadBlocksAsRowsRDDPartition(index: Int, start: Long, end: Long) extends Partition

class ReadBlocksAsRowsRDD(path: String,
  sc: SparkContext,
  partFiles: Array[String],
  partitionCounts: Array[Long],
  gp: GridPartitioner) extends RDD[(Long, Array[Double])](sc, Nil) {
  
  private val partitionStarts = partitionCounts.scanLeft(0L)(_ + _)
  
  if (partitionStarts.last != gp.nRows)
    fatal(s"Error reading BlockMatrix as RowMatrix: expected ${partitionStarts.last} rows in RowMatrix, but found ${gp.nRows} rows in BlockMatrix")

  if (gp.nCols > Int.MaxValue)
      fatal(s"Cannot read BlockMatrix with ${gp.nCols} > Int.MaxValue columns as a RowMatrix")
  
  private val nCols = gp.nCols.toInt
  private val nBlockCols = gp.nBlockCols
  private val blockSize = gp.blockSize

  private val sHadoopBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

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

                val is = sHadoopBc.value.value.unsafeReader(filename)
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
