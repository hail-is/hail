package is.hail.methods

import is.hail.HailContext
import breeze.linalg.SparseVector
import is.hail.utils._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

trait ExportableMatrix {
  def matrix: IndexedRowMatrix
  def hc: HailContext

  def export(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean) {
    exportDelimitedRowSlices(path, columnDelimiter, header, parallelWrite, i => 0, (i, v) => v.length)
  }

  def exportLowerTriangle(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean) {
    exportDelimitedRowSlices(path, columnDelimiter, header, parallelWrite, i => 0, (i, v) => i)
  }

  // includes the diagonal
  def exportStrictLowerTriangle(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean) {
    exportDelimitedRowSlices(path, columnDelimiter, header, parallelWrite, i => 0, (i, v) => i + 1)
  }

  // includes the diagonal
  def exportStrictUpperTriangle(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean) {
    exportDelimitedRowSlices(path, columnDelimiter, header, parallelWrite, i => i, (i, v) => v.length)
  }

  def exportUpperTriangle(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean) {
    exportDelimitedRowSlices(path, columnDelimiter, header, parallelWrite, i => i + 1, (i, v) => v.length)
  }

  // convert elements in [start, end) of each vector into a string, delimited by
  // columnDelimiter, dropping empty rows
  def exportDelimitedRowSlices(path: String, columnDelimiter: String, header: Option[String], parallelWrite: Boolean, start: (Int) => Int, end: (Int, Vector) => Int) {
    genericExport(path, header, parallelWrite, { (sb, i, v) =>
      val l = start(i)
      val r = end(i, v)
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
  def genericExport(path: String, header: Option[String], parallelWrite: Boolean, writeRow: (StringBuilder, Int, Vector) => Unit) {
    prepareMatrixForExport(matrix).rows.mapPartitions { it =>
      val sb = new StringBuilder()
      it.map { (row: IndexedRow) =>
        sb.clear()
        writeRow(sb, row.index.toInt, row.vector)
        sb.result()
      }.filter(s => s.nonEmpty)
    }.writeTable(path, hc.tmpDir, header, parallelWrite)
  }

  /**
    * Creates an IndexedRowMatrix whose backing RDD is sorted by row index and has an entry for every row index.
    *
    * @param matToComplete The matrix to be completed.
    * @return The completed matrix.
    */
  private def prepareMatrixForExport(matToComplete: IndexedRowMatrix): IndexedRowMatrix = {
    val longCols = matrix.numCols()
    require(longCols <= Integer.MAX_VALUE,
      "Cannot export matrices with more than Integer.MAX_VALUE cols")
    val cols = longCols.toInt
    val zeroVector = SparseVector.zeros[Double](cols)
    new IndexedRowMatrix(matToComplete
      .rows
      .map(x => (x.index, x.vector))
      .rightOuterJoin(hc.sc.parallelize(0L until cols).map(x => (x, ())))
      .map {
        case (idx, (Some(v), _)) => IndexedRow(idx, v)
        case (idx, (None, _)) => IndexedRow(idx, zeroVector)
      }.sortBy(_.index))
  }

}
