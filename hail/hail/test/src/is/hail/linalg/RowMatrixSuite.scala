package is.hail.linalg

import is.hail.HailSuite
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.ExportType

import breeze.linalg.DenseMatrix

class RowMatrixSuite extends HailSuite {
  private def rowArrayToRowMatrix(
    a: IndexedSeq[Array[Double]],
    nPartitions: Int = sc.defaultParallelism,
  ): RowMatrix = {
    require(a.nonEmpty)
    val nRows = a.length
    val nCols = a(0).length

    RowMatrix(
      sc.parallelize(
        a.zipWithIndex.map { case (row, i) => (i.toLong, row) },
        nPartitions,
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

  test("localizeRowMatrix") {
    val fname = ctx.createTmpPath("test")

    val rowArrays = ArraySeq(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    val rowMatrix = rowArrayToRowMatrix(rowArrays)
    val localMatrix = rowArrayToLocalMatrix(rowArrays)

    BlockMatrix.fromBreezeMatrix(ctx, localMatrix).write(ctx, fname)

    assertEquals(rowMatrix.toBreezeMatrix(), localMatrix)
  }

  test("readBlockSmall") {
    val fname = ctx.createTmpPath("test")

    val localMatrix = DenseMatrix(
      Array(1.0, 2.0, 3.0),
      Array(4.0, 5.0, 6.0),
    )

    BlockMatrix.fromBreezeMatrix(ctx, localMatrix).write(ctx, fname, forceRowMajor = true)

    val rowMatrixFromBlock = RowMatrix.readBlockMatrix(ctx, fname, 1)

    assertEquals(rowMatrixFromBlock.toBreezeMatrix(), localMatrix)
  }

  test("readBlock") {
    val lm = DenseMatrix.create(9, 10, Array.tabulate(9 * 10)(_.toDouble))
    val fname = ctx.createTmpPath("test")
    cartesian(
      Seq(1, 2, 3, 4, 6, 7, 9, 10),
      Seq(1, 2, 4, 9, 11),
    ).foreach { case (blockSize, partSize) =>
      BlockMatrix.fromBreezeMatrix(ctx, lm, blockSize).write(
        ctx,
        fname,
        overwrite = true,
        forceRowMajor = true,
      )
      val rowMatrix = RowMatrix.readBlockMatrix(ctx, fname, partSize)
      assertEquals(rowMatrix.toBreezeMatrix(), lm)
    }
  }

  private def readCSV(fname: String): Array[Array[Double]] =
    fs.readLines(fname)(it =>
      it.map(_.value)
        .map(_.split(",").map(_.toDouble))
        .toArray[Array[Double]]
    )

  private def exportImportAssert(`export`: (String) => Unit, expected: Array[Double]*): Unit = {
    val fname = ctx.createTmpPath("test")
    `export`(fname)
    assertEquals(
      readCSV(fname).map(_.toSeq).toSeq,
      expected.toArray[Array[Double]].map(_.toSeq).toSeq,
    )
  }

  test("exportWithIndex") {
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

  test("exportSquare") {
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

  test("exportWide") {
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

  test("exportTall") {
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

  test("exportBig") {
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
