package is.hail.linalg

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.ExportType
import is.hail.io.fs.FS

import breeze.linalg.DenseMatrix
import org.junit.jupiter.api.Test

class RowMatrixSuite {
  private def rowArrayToRowMatrix(
    a: IndexedSeq[Array[Double]],
    nPartitions: Int = -1,
  )(implicit ctx: ExecuteContext
  ): RowMatrix = {
    require(a.nonEmpty)
    val nRows = a.length
    val nCols = a(0).length
    val sc = ctx.backend.asSpark.sc
    val np = if (nPartitions == -1) sc.defaultParallelism else nPartitions

    RowMatrix(
      sc.parallelize(
        a.zipWithIndex.map { case (row, i) => (i.toLong, row) },
        np,
      ),
      nCols,
      nRows.toLong,
    )
  }

  private def rowArrayToLocalMatrix(a: IndexedSeq[Array[Double]]): DenseMatrix[Double] = {
    require(a.nonEmpty)
    val nRows = a.length
    val nCols = a(0).length

    new DenseMatrix[Double](nRows, nCols, a.view.flatten.toArray, 0, nCols, isTranspose = true)
  }

  @Test
  def localizeRowMatrix(implicit ctx: ExecuteContext): Unit = {
    val fname = ctx.createTmpPath("test")

    val rowArrays = ArraySeq(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    val localMatrix = rowArrayToLocalMatrix(rowArrays)

    BlockMatrix.fromBreezeMatrix(ctx, localMatrix).write(ctx, fname)

    assertEq(rowMatrix.toBreezeMatrix(), localMatrix)
  }

  @Test
  def readBlockSmall(implicit ctx: ExecuteContext): Unit = {
    val fname = ctx.createTmpPath("test")

    val localMatrix = DenseMatrix(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    BlockMatrix.fromBreezeMatrix(ctx, localMatrix).write(ctx, fname, forceRowMajor = true)

    val rowMatrixFromBlock = RowMatrix.readBlockMatrix(ctx, fname, 1)

    assertEq(rowMatrixFromBlock.toBreezeMatrix(), localMatrix)
  }

  @Test
  def readBlock(implicit ctx: ExecuteContext): Unit = {
    val lm = DenseMatrix.create(9, 10, Array.tabulate(9 * 10)(_.toDouble))
    val fname = ctx.createTmpPath("test")
    cartesian(
      Seq(1, 3, 4, 7, 9, 10),
      Seq(1, 2, 5, 11),
    ).foreach { case (blockSize, partSize) =>
      BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize).write(
        ctx,
        fname,
        overwrite = true,
        forceRowMajor = true,
      )
      val rowMatrix = RowMatrix.readBlockMatrix(ctx, fname, partSize)
      assertEq(rowMatrix.toBreezeMatrix(), lm)
    }
  }

  private def readCSV(fname: String)(implicit fs: FS): Array[Array[Double]] =
    fs.readLines(fname)(it =>
      it.map(_.value)
        .map(_.split(",").map(_.toDouble))
        .toArray[Array[Double]]
    )

  private def exportImportAssert(
    `export`: (String) => Unit,
    expected: Array[Double]*
  )(implicit
    ctx: ExecuteContext
  ): Unit = {
    val fname = ctx.createTmpPath("test")
    `export`(fname)
    val actual = readCSV(fname)(ctx.fs)
    assertEq(actual.map(_.toSeq).toSeq, expected.map(_.toSeq).toSeq)
  }

  @Test
  def exportWithIndex(implicit ctx: ExecuteContext): Unit = {
    val rowArrays = ArraySeq(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
      Array(7.0, 8.0, 9.0),
    )
    val rowMatrix = rowArrayToRowMatrix(rowArrays, nPartitions = 2)

    val rowArraysWithIndex = ArraySeq(
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
  def exportSquare(implicit ctx: ExecuteContext): Unit = {
    val rowArrays = ArraySeq(
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
  def exportWide(implicit ctx: ExecuteContext): Unit = {
    val rowArrays = ArraySeq(
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
  def exportTall(implicit ctx: ExecuteContext): Unit = {
    val rowArrays = ArraySeq(
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
  def exportBig(implicit ctx: ExecuteContext): Unit = {
    val rowArrays: ArraySeq[Array[Double]] =
      ArraySeq.tabulate(20)(r => Array.tabulate(30)(c => 30.0 * c + r))
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
        }: _*
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
        .filter(_.nonEmpty): _*
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
        }: _*
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
        .filter(_.nonEmpty): _*
    )
  }
}
