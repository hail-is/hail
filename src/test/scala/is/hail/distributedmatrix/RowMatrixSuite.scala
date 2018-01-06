package is.hail.distributedmatrix
  
import is.hail.SparkSuite
import is.hail.utils._
import org.testng.annotations.Test


class RowMatrixSuite extends SparkSuite {
  private def readCSV(fname: String): Array[Array[Double]] =
    hc.hadoopConf.readLines(fname)( it =>
      it.map(_.value)
        .map(_.split(",").map(_.toDouble))
        .toArray[Array[Double]]
    )

  private def exportImportAssert(export: (String) => Unit, expected: Array[Double]*) {
    val fname = tmpDir.createTempFile("test")
    export(fname)
    assert(readCSV(fname) === expected.toArray[Array[Double]])
  }

  private def rowArrayToRowMatrix(a: Array[Array[Double]]): RowMatrix = {
    val nRows = a.length
    val nCols = if (nRows == 0) 0 else a(0).length
    RowMatrix(sc.parallelize(a.zipWithIndex.map { case (a, i) => (i, a) }), nCols, nRows)
  }

  @Test
  def exportSquare() {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(rowMatrix.export(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0, 9.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(4.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0),
      Array(9.0))

    exportImportAssert(rowMatrix.exportStrictUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(2.0, 3.0),
      Array(6.0))
  }
  
  @Test
  def exportWide() {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(rowMatrix.export(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(4.0))
    
    exportImportAssert(rowMatrix.exportUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0))
    
    exportImportAssert(rowMatrix.exportStrictUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(2.0, 3.0),
      Array(6.0))
  }
  
  @Test
  def exportTall() {
    val rowArrays = Array(
      Array(1.0, 2.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(rowMatrix.export(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(4.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0),
      Array(5.0))
    
    exportImportAssert(rowMatrix.exportStrictUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      Array(2.0))
  }  

  @Test
  def exportBigish() {
    val rowArrays: Array[Array[Double]] = Array.tabulate(20)( r => Array.tabulate(30)(c => 30 * c + r))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    
    exportImportAssert(rowMatrix.export(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j <= i }.map(_._1) }
        .toArray[Array[Double]]:_*)
        
    exportImportAssert(rowMatrix.exportStrictLowerTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j < i }.map(_._1) }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)

    exportImportAssert(rowMatrix.exportUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j >= i }.map(_._1) }
        .toArray[Array[Double]]:_*)

    exportImportAssert(rowMatrix.exportStrictUpperTriangle(hc, _, ",", header=None, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j > i }.map(_._1) }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)
  }
}