package is.hail.linalg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr.ir.IRParser
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

  def copyRowMajorToColumnMajor(rowMajorFirstElementAddress: Long, targetFirstElementAddress: Long, nRows: Long, nCols: Long, elementPTypeString: String): Unit = {
    val elementPType = IRParser.parsePType(elementPTypeString)

    var rowIndex = 0L
    while(rowIndex < nRows) {
      var colIndex = 0L
      while(colIndex < nCols) {
        val rowMajorCoord = nCols * rowIndex + colIndex
        val colMajorCoord = nRows * colIndex + rowIndex
        val currentElement = Region.loadPrimitiveUnstaged(elementPType)(rowMajorFirstElementAddress + rowMajorCoord * elementPType.byteSize)
        val targetAddress = targetFirstElementAddress + colMajorCoord * elementPType.byteSize
        Region.storePrimitiveUnstaged(elementPType, targetAddress)(currentElement)
        colIndex += 1
      }
      rowIndex += 1
    }
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

  def copyColumnMajorToRowMajor(colMajorFirstElementAddress: Long, targetFirstElementAddress: Long, nRows: Long, nCols: Long, elementPTypeString: String): Unit = {
    val elementPType = IRParser.parsePType(elementPTypeString)

    var rowIndex = 0L
    while (rowIndex < nRows) {
      var colIndex = 0L
      while (colIndex < nCols) {
        val rowMajorCoord = nCols * rowIndex + colIndex
        val colMajorCoord = nRows * colIndex + rowIndex
        val currentElement = Region.loadPrimitiveUnstaged(elementPType)(colMajorFirstElementAddress + colMajorCoord * elementPType.byteSize)
        val targetAddress = targetFirstElementAddress + rowMajorCoord * elementPType.byteSize
        Region.storePrimitiveUnstaged(elementPType, targetAddress)(currentElement)
        colIndex += 1
      }
      rowIndex += 1
    }
  }
}
