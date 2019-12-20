package is.hail.linalg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr.types.physical.PType

object LinalgCodeUtils {
  def copyRowMajorToColumnMajor(rowMajorFirstElementAddress: Code[Long], targetFirstElementAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], elementPType: PType, mb: MethodBuilder): Code[Unit] = {
    val rowIndex = mb.newField[Long]
    val colIndex = mb.newField[Long]
    val rowMajorCoord: Code[Long] = nCols * rowIndex + colIndex
    val colMajorCoord: Code[Long] = nRows * colIndex + rowIndex
    val currentElement = Region.loadPrimitive(elementPType)(rowMajorFirstElementAddress + rowMajorCoord * elementPType.byteSize)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storePrimitive(elementPType, (targetFirstElementAddress + colMajorCoord * elementPType.byteSize))(currentElement)
      )
    )
  }

  def copyColumnMajorToRowMajor(colMajorFirstElementAddress: Code[Long], targetFirstElementAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], elementPType: PType, mb: MethodBuilder): Code[Unit] = {
    val rowIndex = mb.newField[Long]
    val colIndex = mb.newField[Long]
    val rowMajorCoord: Code[Long] = nCols * rowIndex + colIndex
    val colMajorCoord: Code[Long] = nRows * colIndex + rowIndex
    val currentElement = Region.loadPrimitive(elementPType)(colMajorFirstElementAddress + colMajorCoord * elementPType.byteSize)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storePrimitive(elementPType, targetFirstElementAddress + rowMajorCoord * elementPType.byteSize)(currentElement)
      )
    )
  }
}
