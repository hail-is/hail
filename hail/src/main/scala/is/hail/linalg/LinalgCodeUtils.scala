package is.hail.linalg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.physical.stypes.concrete.{SNDArrayPointer, SNDArrayPointerSettable}
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayCode, SNDArrayValue}

object LinalgCodeUtils {
  def checkColumnMajor(pndv: SNDArrayValue, cb: EmitCodeBuilder): Value[Boolean] = {
    val answer = cb.newField[Boolean]("checkColumnMajorResult")
    val shapes = pndv.shapes(cb)
    val strides = pndv.strides(cb)
    val runningProduct = cb.newLocal[Long]("check_column_major_running_product")

    val pt = pndv.st.asInstanceOf[SNDArrayPointer].pType
    val elementType = pt.elementType
    val nDims = pndv.st.nDims

    cb.assign(answer, true)
    cb.assign(runningProduct, elementType.byteSize)
    (0 until nDims).foreach{ index =>
      cb.assign(answer, answer & (strides(index) ceq runningProduct))
      cb.assign(runningProduct, runningProduct * (shapes(index) > 0L).mux(shapes(index), 1L))
    }
    answer
  }

  def checkRowMajor(pndv: SNDArrayValue, cb: EmitCodeBuilder): Value[Boolean] = {
    val answer = cb.newField[Boolean]("checkColumnMajorResult")
    val shapes = pndv.shapes(cb)
    val strides = pndv.strides(cb)
    val runningProduct = cb.newLocal[Long]("check_column_major_running_product")

    val pt = pndv.st.asInstanceOf[SNDArrayPointer].pType
    val elementType = pt.elementType
    val nDims = pt.nDims

    cb.assign(answer, true)
    cb.assign(runningProduct, elementType.byteSize)
    ((nDims - 1) to 0 by -1).foreach { index =>
      cb.assign(answer, answer & (strides(index) ceq runningProduct))
      cb.assign(runningProduct, runningProduct * (shapes(index) > 0L).mux(shapes(index), 1L))
    }
    answer
  }

  def createColumnMajorCode(pndv: SNDArrayValue, cb: EmitCodeBuilder, region: Value[Region]): SNDArrayCode = {
    val shape = pndv.shapes(cb)
    val pt = pndv.st.asInstanceOf[SNDArrayPointer].pType
    val strides = pt.makeColumnMajorStrides(shape, region, cb)

    val (dataFirstElementAddress, dataFinisher) = pt.constructDataFunction(shape, strides, cb, region)

    val curAddr = cb.newLocal[Long]("create_column_major_cur_addr", dataFirstElementAddress)

    SNDArray.forEachIndex(cb, shape, "nda_create_column_major") { case (cb, idxVars) =>
      pt.elementType.storeAtAddress(cb, curAddr, region, pndv.loadElement(idxVars, cb), true)
      cb.assign(curAddr, curAddr + pt.elementType.byteSize)
    }
    dataFinisher(cb)
  }

  def checkColMajorAndCopyIfNeeded(aInput: SNDArrayValue, cb: EmitCodeBuilder, region: Value[Region]): SNDArrayValue = {
    val aIsColumnMajor = LinalgCodeUtils.checkColumnMajor(aInput, cb)
    val aColMajor = cb.emb.newPField("ndarray_output_column_major", aInput.st).asInstanceOf[SNDArrayPointerSettable]
    cb.ifx(aIsColumnMajor, {cb.assign(aColMajor, aInput)},
      {
        cb.assign(aColMajor, LinalgCodeUtils.createColumnMajorCode(aInput, cb, region))
      })
    aColMajor
  }

  def checkStandardStriding(aInput: SNDArrayValue, cb: EmitCodeBuilder, region: Value[Region]): (SNDArrayValue, Value[Boolean]) = {
    val aIsColumnMajor = LinalgCodeUtils.checkColumnMajor(aInput, cb)
    val a = cb.emb.newPField("ndarray_output_standardized", aInput.st).asInstanceOf[SNDArrayPointerSettable]
    cb.ifx(aIsColumnMajor, {cb.assign(a, aInput)}, {
      val isRowMajor = LinalgCodeUtils.checkRowMajor(aInput, cb)
      cb.ifx(isRowMajor, {cb.assign(a, aInput)}, {
        cb.assign(a, LinalgCodeUtils.createColumnMajorCode(aInput, cb, region))
      })
    })

    (a, aIsColumnMajor)
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
