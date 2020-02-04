package is.hail.linalg

import is.hail.annotations.Region
import is.hail.annotations.Region.{loadBoolean, loadDouble, loadFloat, loadLong}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.utils._
import is.hail.asm4s._
import is.hail.expr.ir.IRParser
import is.hail.expr.types.physical.{PBoolean, PFloat32, PFloat64, PInt32, PInt64, PType}

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

  def loadPrimitiveUnstaged(typ: PType): Long => AnyVal = typ.fundamentalType match {
    case _: PBoolean => loadBoolean
    case _: PInt32 => loadDouble
    case _: PInt64 => loadLong
    case _: PFloat32 => loadFloat
    case _: PFloat64 => loadDouble
  }

  def storePrimitiveUnstaged(address: Long, value: AnyVal) = value match {
    case x: Boolean => Region.storeBoolean(address, x)
    case x: Int => Region.storeInt(address, x)
    case x: Long => Region.storeLong(address, x)
    case x: Float => Region.storeFloat(address, x)
    case x: Double => Region.storeDouble(address, x)
  }

  def copyRowMajorToColumnMajor(rowMajorFirstElementAddress: Long, targetFirstElementAddress: Long, nRows: Long, nCols: Long, elementPTypeString: String): Unit = {
    val elementPType = IRParser.parsePType(elementPTypeString)

    var rowIndex = 0L
    while(rowIndex < nRows) {
      var colIndex = 0L
      while(colIndex < nCols) {
        val rowMajorCoord = nCols * rowIndex + colIndex
        val colMajorCoord = nRows * colIndex + rowIndex
        val currentElement = loadPrimitiveUnstaged(elementPType)(rowMajorFirstElementAddress + rowMajorCoord * elementPType.byteSize)
        val targetAddress = targetFirstElementAddress + colMajorCoord * elementPType.byteSize
        storePrimitiveUnstaged(targetAddress, currentElement)
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
}
