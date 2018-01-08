package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.utils._
import org.apache.spark.rdd.RDD

object RowMatrix {
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int): RowMatrix =
    new RowMatrix(hc, rows, nCols, None, None)
  
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long): RowMatrix =
    new RowMatrix(hc, rows, nCols, Some(nRows), None)
  
  def apply(hc: HailContext, rows: RDD[(Long, Array[Double])], nCols: Int, nRows: Long, partitionCounts: Array[Long]): RowMatrix =
    new RowMatrix(hc, rows, nCols, Some(nRows), Some(partitionCounts))
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
    
  def export(path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, exportType, _ => 0, _ => localNCols)
  }

  // includes the diagonal
  def exportLowerTriangle(path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, exportType, _ => 0, i => math.min(i + 1, localNCols.toLong).toInt)
  }

  def exportStrictLowerTriangle(path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, exportType, _ => 0, i => math.min(i, localNCols.toLong).toInt)
  }

  // includes the diagonal
  def exportUpperTriangle(path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, exportType, i => math.min(i, localNCols.toLong).toInt, _ => localNCols)
  }  
    
  def exportStrictUpperTriangle(path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    val localNCols = nCols
    exportDelimitedRowSlices(path, columnDelimiter, header, exportType, i => math.min(i + 1, localNCols.toLong).toInt, _ => localNCols)
  }
  
  // convert elements in [start, end) of each array to a string, delimited by columnDelimiter, and export
  def exportDelimitedRowSlices(
    path: String, 
    columnDelimiter: String,
    header: Option[String], 
    exportType: Int, 
    start: (Long) => Int, 
    end: (Long) => Int) {
    
    genericExport(path, header, exportType, { (sb, i, v) =>
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