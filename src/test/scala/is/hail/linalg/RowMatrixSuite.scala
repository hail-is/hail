package is.hail.linalg
  
import breeze.linalg.DenseMatrix
import is.hail.SparkSuite
import is.hail.check.Gen
import is.hail.utils._
import org.testng.annotations.Test

class RowMatrixSuite extends SparkSuite {
  private def rowArrayToRowMatrix(a: Array[Array[Double]], nPartitions: Int = sc.defaultParallelism): RowMatrix = {
    require(a.length > 0)
    val nRows = a.length
    val nCols = a(0).length
    
    RowMatrix(hc, sc.parallelize(a.zipWithIndex.map { case (row, i) => (i.toLong, row) }, nPartitions), nCols, nRows)
  }
  
  private def rowArrayToLocalMatrix(a: Array[Array[Double]]): DenseMatrix[Double] = {
    require(a.length > 0)
    val nRows = a.length
    val nCols = a(0).length
    
    new DenseMatrix[Double](nRows, nCols, a.flatten, 0, nCols, isTranspose = true)
  }
  
  @Test
  def localizeRowMatrix() {
    val fname = tmpDir.createTempFile("test")
    
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0))

    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    val localMatrix = rowArrayToLocalMatrix(rowArrays)
    
    BlockMatrix.fromBreezeMatrix(hc.sc, localMatrix).write(fname)
    
    assert(rowMatrix.toBreezeMatrix() === localMatrix)
  }

  @Test
  def readBlockSmall() {
    val fname = tmpDir.createTempFile("test")
    
    val localMatrix = DenseMatrix(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0))
    
    BlockMatrix.fromBreezeMatrix(hc.sc, localMatrix).write(fname, forceRowMajor = true)
    
    val rowMatrixFromBlock = RowMatrix.readBlockMatrix(hc, fname, Some(1))
    
    assert(rowMatrixFromBlock.toBreezeMatrix() == localMatrix)
  }
  
  @Test
  def readBlock() {
    val fname = tmpDir.createTempFile("test")    
    val lm = Gen.denseMatrix[Double](9, 10).sample()

    for {
      blockSize <- Seq(1, 2, 3, 4, 6, 7, 9, 10)
      partSize <- Seq(1, 2, 4, 9, 11)
    } {
      BlockMatrix.fromBreezeMatrix(sc, lm, blockSize).write(fname, forceRowMajor = true)
      val rowMatrix = RowMatrix.readBlockMatrix(hc, fname, Some(partSize))
      
      assert(rowMatrix.toBreezeMatrix() === lm)
    }
  }
  
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

  @Test
  def exportWithIndex() {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays, nPartitions = 2)

    val rowArraysWithIndex = Array(
      Array(0.0, 1.0, 2.0, 3.0),
      Array(1.0, 4.0, 5.0, 6.0),
      Array(2.0, 7.0, 8.0, 9.0))

    exportImportAssert(rowMatrix.export(_, ",", header = None, addIndex = true, exportType = ExportType.CONCATENATED),
      rowArraysWithIndex: _*)
  }

  @Test
  def exportSquare() {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(rowMatrix.export(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0, 9.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(4.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0),
      Array(9.0))

    exportImportAssert(rowMatrix.exportStrictUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(2.0, 3.0),
      Array(6.0))
  }
  
  @Test
  def exportWide() {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(rowMatrix.export(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(4.0))
    
    exportImportAssert(rowMatrix.exportUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0))
    
    exportImportAssert(rowMatrix.exportStrictUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
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

    exportImportAssert(rowMatrix.export(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportStrictLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(4.0),
      Array(7.0, 8.0))

    exportImportAssert(rowMatrix.exportUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(1.0, 2.0),
      Array(5.0))
    
    exportImportAssert(rowMatrix.exportStrictUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      Array(2.0))
  }  

  @Test
  def exportBig() {
    val rowArrays: Array[Array[Double]] = Array.tabulate(20)( r => Array.tabulate(30)(c => 30 * c + r))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    
    exportImportAssert(rowMatrix.export(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays: _*)

    exportImportAssert(rowMatrix.exportLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j <= i }.map(_._1) }
        .toArray[Array[Double]]:_*)
        
    exportImportAssert(rowMatrix.exportStrictLowerTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j < i }.map(_._1) }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)

    exportImportAssert(rowMatrix.exportUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j >= i }.map(_._1) }
        .toArray[Array[Double]]:_*)

    exportImportAssert(rowMatrix.exportStrictUpperTriangle(_, ",", header=None, addIndex = false, exportType = ExportType.CONCATENATED),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j > i }.map(_._1) }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]:_*)
  }
}