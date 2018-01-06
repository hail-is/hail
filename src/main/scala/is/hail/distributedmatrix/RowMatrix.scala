package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.utils._
import org.apache.spark.rdd.RDD

object RowMatrix {
  def apply(rows: RDD[(Int, Array[Double])], nCols: Int): RowMatrix =
    new RowMatrix(rows, nCols, None, None)
  
  def apply(rows: RDD[(Int, Array[Double])], nCols: Int, nRows: Long): RowMatrix =
    new RowMatrix(rows, nCols, Some(nRows), None)
  
  def apply(rows: RDD[(Int, Array[Double])], nCols: Int, nRows: Long, partitionCounts: Array[Long]): RowMatrix =
    new RowMatrix(rows, nCols, Some(nRows), Some(partitionCounts))
}

class RowMatrix(val rows: RDD[(Int, Array[Double])],
  val nCols: Int,
  private var _nRows: Option[Long],
  private var _partitionCounts: Option[Array[Long]]) extends Serializable {

  def nRows: Long = {
    if (_nRows.isEmpty)
      _nRows = Some(partitionCounts().sum)
    _nRows.get
  }
  
  def partitionCounts(): Array[Long] = {
    if (_partitionCounts.isEmpty)
      _partitionCounts = Some(rows.countPerPartition())
    _partitionCounts.get
  }
  
  // length nPartitions + 1, first element 0, last element rdd2 count
  def partitionStarts(): Array[Long] = partitionCounts().scanLeft(0L)(_ + _)
  
  def export(hc: HailContext, path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    exportDelimitedRowSlices(hc, path, columnDelimiter, header, exportType, i => 0, i => nCols)
  }

  // includes the diagonal
  def exportLowerTriangle(hc: HailContext, path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    exportDelimitedRowSlices(hc, path, columnDelimiter, header, exportType, i => 0, i => math.min(i + 1, nCols))
  }

  def exportStrictLowerTriangle(hc: HailContext, path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    exportDelimitedRowSlices(hc, path, columnDelimiter, header, exportType, i => 0, i => math.min(i, nCols))
  }
    
  def exportStrictUpperTriangle(hc: HailContext, path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    exportDelimitedRowSlices(hc, path, columnDelimiter, header, exportType, i => i + 1, i => nCols)
  }

  // includes the diagonal
  def exportUpperTriangle(hc: HailContext, path: String, columnDelimiter: String, header: Option[String], exportType: Int) {
    exportDelimitedRowSlices(hc, path, columnDelimiter, header, exportType, i => i, i => nCols)
  }  
  
  // convert elements in [start, end) of each vector into a string, delimited by
  // columnDelimiter, dropping empty rows
  def exportDelimitedRowSlices(hc: HailContext,
    path: String, 
    columnDelimiter: String,
    header: Option[String], 
    exportType: Int, 
    start: (Int) => Int, 
    end: (Int) => Int) {
    
    genericExport(hc, path, header, exportType, { (sb, i, v) =>
      val l = start(i)
      val r = end(i)
      var j = l
      while (j < r) {
        if (j > l)
          sb ++= columnDelimiter
        sb.append(v(j))
        j += 1
      }
    })
  }

  // uses writeRow to convert each row to a String and writes that to a file,
  // unless the String is empty
  def genericExport(hc: HailContext,
    path: String, 
    header: Option[String], 
    exportType: Int, 
    writeRow: (StringBuilder, Int, Array[Double]) => Unit) {
    
    rows.mapPartitions { it =>
      val sb = new StringBuilder()
      it.map { case (index, data) =>
        sb.clear()
        writeRow(sb, index.toInt, data)
        sb.result()
      }.filter(s => s.nonEmpty)
    }.writeTable(path, hc.tmpDir, header, exportType)
  }
}