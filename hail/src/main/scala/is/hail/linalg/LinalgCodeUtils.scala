package is.hail.linalg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.asm4s._
import is.hail.utils._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.{PBaseStructValue, PCode, PNDArrayCode, PNDArrayValue, PType, typeToTypeInfo}

object LinalgCodeUtils {
  def checkColumnMajor(pndv: PNDArrayValue, cb: EmitCodeBuilder): Value[Boolean] = {
    val answer = cb.newField[Boolean]("checkColumnMajorResult")
    val shapes = pndv.shapes()
    val runningProduct = cb.newLocal[Long]("check_column_major_running_product")

    val elementType = pndv.pt.elementType
    val nDims = pndv.pt.nDims
    cb.append(Code(
      runningProduct := elementType.byteSize,
      Code.foreach(0 until nDims){ index =>
        Code(
          answer := answer & (shapes(index) ceq runningProduct),
          runningProduct := runningProduct * (shapes(index) > 0L).mux(shapes(index), 1L)
        )
      }
    ))
    answer
  }

  def createColumnMajorCode(pndv: PNDArrayValue, cb: EmitCodeBuilder, region: Value[Region]): PNDArrayCode = {
    val shape = pndv.shapes()
    val shapeBuilder = pndv.pt.makeShapeBuilder(shape)
    val stridesBuilder = pndv.pt.makeColumnMajorStridesBuilder(shape, cb.emb)
    val dataLength = pndv.pt.numElements(shape, cb.emb)

    val outputElementPType = pndv.pt.elementType
    val idxVars = Array.tabulate(pndv.pt.nDims) { _ => cb.emb.genFieldThisRef[Long]() }.toFastIndexedSeq

    def loadElement(ndValue: PNDArrayValue) = {
      ndValue.pt.loadElementToIRIntermediate(idxVars, ndValue.value.asInstanceOf[Value[Long]], cb.emb)
    }

    val srvb = new StagedRegionValueBuilder(cb.emb, pndv.pt.data.pType, region)

    val body =
      Code(
        srvb.addIRIntermediate(outputElementPType)(loadElement(pndv)),
        srvb.advance()
      )

    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(body) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < shape(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    cb.append(Code(
      srvb.start(dataLength.toI),
      columnMajorLoops
    ))

    val answer = pndv.pt.construct(shapeBuilder, stridesBuilder, srvb.end(), cb.emb, region)
    PCode.apply(pndv.pt, answer).asNDArray
  }

  def copyRowMajorToColumnMajor(rowMajorFirstElementAddress: Code[Long], targetFirstElementAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], elementPType: PType, mb: MethodBuilder[_]): Code[Unit] = {
    val rowIndex = mb.genFieldThisRef[Long]()
    val colIndex = mb.genFieldThisRef[Long]()
    val rowMajorCoord: Code[Long] = nCols * rowIndex + colIndex
    val colMajorCoord: Code[Long] = nRows * colIndex + rowIndex
    val currentElement = Region.loadPrimitive(elementPType)(rowMajorFirstElementAddress + rowMajorCoord * elementPType.byteSize)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storePrimitive(elementPType, (targetFirstElementAddress + colMajorCoord * elementPType.byteSize))(currentElement)
      )
    )
  }

  def copyRowMajorToColumnMajor(rowMajorFirstElementAddress: Long, targetFirstElementAddress: Long, nRows: Long, nCols: Long, elementByteSize: Long): Unit = {

    var rowIndex = 0L
    while(rowIndex < nRows) {
      var colIndex = 0L
      while(colIndex < nCols) {
        val rowMajorCoord = nCols * rowIndex + colIndex
        val colMajorCoord = nRows * colIndex + rowIndex
        val sourceAddress = rowMajorFirstElementAddress + rowMajorCoord * elementByteSize
        val targetAddress = targetFirstElementAddress + colMajorCoord * elementByteSize
        Region.copyFrom(sourceAddress, targetAddress, elementByteSize)
        colIndex += 1
      }
      rowIndex += 1
    }
  }

  def copyColumnMajorToRowMajor(colMajorFirstElementAddress: Code[Long], targetFirstElementAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], elementPType: PType, mb: MethodBuilder[_]): Code[Unit] = {
    val rowIndex = mb.genFieldThisRef[Long]()
    val colIndex = mb.genFieldThisRef[Long]()
    val rowMajorCoord: Code[Long] = nCols * rowIndex + colIndex
    val colMajorCoord: Code[Long] = nRows * colIndex + rowIndex
    val currentElement = Region.loadPrimitive(elementPType)(colMajorFirstElementAddress + colMajorCoord * elementPType.byteSize)

    Code.forLoop(rowIndex := 0L, rowIndex < nRows, rowIndex := rowIndex + 1L,
      Code.forLoop(colIndex := 0L, colIndex < nCols, colIndex := colIndex + 1L,
        Region.storePrimitive(elementPType, targetFirstElementAddress + rowMajorCoord * elementPType.byteSize)(currentElement)
      )
    )
  }

  def copyColumnMajorToRowMajor(colMajorFirstElementAddress: Long, targetFirstElementAddress: Long, nRows: Long, nCols: Long, elementByteSize: Long): Unit = {
    var rowIndex = 0L
    while (rowIndex < nRows) {
      var colIndex = 0L
      while (colIndex < nCols) {
        val rowMajorCoord = nCols * rowIndex + colIndex
        val colMajorCoord = nRows * colIndex + rowIndex
        val sourceAddress = colMajorFirstElementAddress + colMajorCoord * elementByteSize
        val targetAddress = targetFirstElementAddress + rowMajorCoord * elementByteSize
        Region.copyFrom(sourceAddress, targetAddress, elementByteSize)
        colIndex += 1
      }
      rowIndex += 1
    }
  }
}
