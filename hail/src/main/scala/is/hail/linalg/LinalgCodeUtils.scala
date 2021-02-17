package is.hail.linalg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.physical.stypes.interfaces.SNDArray
import is.hail.types.physical.{PCanonicalNDArray, PNDArrayCode, PNDArrayValue}

object LinalgCodeUtils {
  def checkColumnMajor(pndv: PNDArrayValue, cb: EmitCodeBuilder): Value[Boolean] = {
    val answer = cb.newField[Boolean]("checkColumnMajorResult")
    val shapes = pndv.shapes(cb)
    val strides = pndv.strides(cb)
    val runningProduct = cb.newLocal[Long]("check_column_major_running_product")

    val elementType = pndv.pt.elementType
    val nDims = pndv.pt.nDims

    cb.assign(answer, true)
    cb.append(Code(
      runningProduct := elementType.byteSize,
      Code.foreach(0 until nDims){ index =>
        Code(
          answer := answer & (strides(index) ceq runningProduct),
          runningProduct := runningProduct * (shapes(index) > 0L).mux(shapes(index), 1L)
        )
      }
    ))
    answer
  }

  def createColumnMajorCode(pndv: PNDArrayValue, cb: EmitCodeBuilder, region: Value[Region]): PNDArrayCode = {
    val shape = pndv.shapes(cb)
    val pt = pndv.pt.asInstanceOf[PCanonicalNDArray]
    val strides = pt.makeColumnMajorStrides(shape, region, cb)

    val (dataFirstElementAddress, dataFinisher) = pndv.pt.constructDataFunction(shape, strides, cb, region)

    val curAddr = cb.newLocal[Long]("create_column_major_cur_addr", dataFirstElementAddress)

    SNDArray.forEachIndex(cb, shape, "nda_create_column_major") { case (cb, idxVars) =>
      pt.elementType.storeAtAddress(cb, curAddr, region, pndv.loadElement(idxVars, cb), true)
      cb.assign(curAddr, curAddr + pt.elementType.byteSize)
    }
    dataFinisher(cb)
  }

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = {
    val index = mb.genFieldThisRef[Long]()
    val elementsInProcessedDimensions = mb.genFieldThisRef[Long]()
    Code(
      index := 0L,
      elementsInProcessedDimensions := 1L,
      Code.foreach(shapeArray.zip(indices).reverse) { case (shapeElement, currentIndex) =>
        Code(
          index := index + currentIndex * elementsInProcessedDimensions,
          elementsInProcessedDimensions := elementsInProcessedDimensions * shapeElement
        )
      },
      index
    )
  }

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): (Code[Unit], IndexedSeq[Value[Long]]) = {
    val nDim = shapeArray.length
    val newIndices = (0 until nDim).map(_ => mb.genFieldThisRef[Long]())
    val elementsInProcessedDimensions = mb.genFieldThisRef[Long]()
    val workRemaining = mb.genFieldThisRef[Long]()

    val createShape = Code(
      workRemaining := index,
      elementsInProcessedDimensions := shapeArray.foldLeft(1L: Code[Long])(_ * _),
      Code.foreach(shapeArray.zip(newIndices)) { case (shapeElement, newIndex) =>
        Code(
          elementsInProcessedDimensions := elementsInProcessedDimensions / shapeElement,
          newIndex := workRemaining / elementsInProcessedDimensions,
          workRemaining := workRemaining % elementsInProcessedDimensions
        )
      }
    )
    (createShape, newIndices)
  }
}
