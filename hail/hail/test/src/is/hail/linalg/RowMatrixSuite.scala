package is.hail.linalg

import is.hail.HailSuite
import is.hail.scalacheck._
import is.hail.utils._

import breeze.linalg.DenseMatrix
import org.scalatest
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class RowMatrixSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {
  private def rowArrayToRowMatrix(a: Array[Array[Double]], nPartitions: Int = sc.defaultParallelism)
    : RowMatrix = {
    require(a.length > 0)
    val nRows = a.length
    val nCols = a(0).length

    RowMatrix(
      sc.parallelize(a.zipWithIndex.map { case (row, i) => (i.toLong, row) }, nPartitions),
      nCols,
      nRows,
    )
  }

  private def rowArrayToLocalMatrix(a: Array[Array[Double]]): DenseMatrix[Double] = {
    require(a.length > 0)
    val nRows = a.length
    val nCols = a(0).length

    new DenseMatrix[Double](nRows, nCols, a.flatten, 0, nCols, isTranspose = true)
  }

  @Test
  def localizeRowMatrix(): scalatest.Assertion = {
    val fname = ctx.createTmpPath("test")

    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    val localMatrix = rowArrayToLocalMatrix(rowArrays)

    BlockMatrix.fromBreezeMatrix(localMatrix).write(ctx, fname)

    assert(rowMatrix.toBreezeMatrix() === localMatrix)
  }

  @Test
  def readBlockSmall(): scalatest.Assertion = {
    val fname = ctx.createTmpPath("test")

    val localMatrix = DenseMatrix(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    BlockMatrix.fromBreezeMatrix(localMatrix).write(ctx, fname, forceRowMajor = true)

    val rowMatrixFromBlock = RowMatrix.readBlockMatrix(fs, fname, 1)

    assert(rowMatrixFromBlock.toBreezeMatrix() == localMatrix)
  }

  @Test
  def readBlock(): scalatest.Assertion =
    forAll(genDenseMatrix(9, 10)) { lm =>
      val fname = ctx.createTmpPath("test")
      scalatest.Inspectors.forAll {
        cartesian(
          Seq(1, 2, 3, 4, 6, 7, 9, 10),
          Seq(1, 2, 4, 9, 11),
        )
      } { case (blockSize, partSize) =>
        BlockMatrix.fromBreezeMatrix(lm, blockSize).write(
          ctx,
          fname,
          overwrite = true,
          forceRowMajor = true,
        )
        val rowMatrix = RowMatrix.readBlockMatrix(fs, fname, partSize)
        assert(rowMatrix.toBreezeMatrix() === lm)
      }
    }

  private def readCSV(fname: String): Array[Array[Double]] =
    fs.readLines(fname)(it =>
      it.map(_.value)
        .map(_.split(",").map(_.toDouble))
        .toArray[Array[Double]]
    )

  private def exportImportAssert(export: (String) => Unit, expected: Array[Double]*)
    : scalatest.Assertion = {
    val fname = ctx.createTmpPath("test")
    export(fname)
    assert(readCSV(fname) === expected.toArray[Array[Double]])
  }

  @Test
  def exportWithIndex(): scalatest.Assertion = {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0),
    )
    val rowMatrix = rowArrayToRowMatrix(rowArrays, nPartitions = 2)

    val rowArraysWithIndex = Array(
      Array(0.0, 1.0, 2.0, 3.0),
      Array(1.0, 4.0, 5.0, 6.0),
      Array(2.0, 7.0, 8.0, 9.0),
    )

    exportImportAssert(
      rowMatrix.export(
        ctx,
        _,
        ",",
        header = None,
        addIndex = true,
        exportType = ExportType.CONCATENATED,
      ),
      rowArraysWithIndex: _*
    )
  }

  @Test
  def exportSquare(): scalatest.Assertion = {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0),
    )
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(
      rowMatrix.export(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays: _*
    )

    exportImportAssert(
      rowMatrix.exportLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0, 9.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(4.0),
      Array(7.0, 8.0),
    )

    exportImportAssert(
      rowMatrix.exportUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0),
      Array(9.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(2.0, 3.0),
      Array(6.0),
    )
  }

  @Test
  def exportWide(): scalatest.Assertion = {
    val rowArrays = Array(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(
      rowMatrix.export(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays: _*
    )

    exportImportAssert(
      rowMatrix.exportLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0),
      Array(4.0, 5.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(4.0),
    )

    exportImportAssert(
      rowMatrix.exportUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0, 2.0, 3.0),
      Array(5.0, 6.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(2.0, 3.0),
      Array(6.0),
    )
  }

  @Test
  def exportTall(): scalatest.Assertion = {
    val rowArrays = Array(
      Array(1.0, 2.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0),
    )
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(
      rowMatrix.export(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays: _*
    )

    exportImportAssert(
      rowMatrix.exportLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0),
      Array(4.0, 5.0),
      Array(7.0, 8.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(4.0),
      Array(7.0, 8.0),
    )

    exportImportAssert(
      rowMatrix.exportUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(1.0, 2.0),
      Array(5.0),
    )

    exportImportAssert(
      rowMatrix.exportStrictUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      Array(2.0),
    )
  }

  @Test
  def exportBig(): scalatest.Assertion = {
    val rowArrays: Array[Array[Double]] =
      Array.tabulate(20)(r => Array.tabulate(30)(c => 30 * c + r))
    val rowMatrix = rowArrayToRowMatrix(rowArrays)

    exportImportAssert(
      rowMatrix.export(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays: _*
    )

    exportImportAssert(
      rowMatrix.exportLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j <= i }.map(_._1)
        }
        .toArray[Array[Double]]: _*
    )

    exportImportAssert(
      rowMatrix.exportStrictLowerTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j < i }.map(_._1)
        }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]: _*
    )

    exportImportAssert(
      rowMatrix.exportUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j >= i }.map(_._1)
        }
        .toArray[Array[Double]]: _*
    )

    exportImportAssert(
      rowMatrix.exportStrictUpperTriangle(
        ctx,
        _,
        ",",
        header = None,
        addIndex = false,
        exportType = ExportType.CONCATENATED,
      ),
      rowArrays.zipWithIndex
        .map { case (a, i) =>
          a.zipWithIndex.filter { case (_, j) => j > i }.map(_._1)
        }
        .filter(_.nonEmpty)
        .toArray[Array[Double]]: _*
    )
  }
}
