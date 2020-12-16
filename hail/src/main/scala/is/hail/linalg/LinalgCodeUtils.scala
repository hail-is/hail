package is.hail.linalg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.asm4s._
import is.hail.utils._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.{PBaseStructValue, PCode, PNDArrayCode, PNDArrayValue, PType, typeToTypeInfo}

object LinalgCodeUtils {
  def checkColumnMajor(pndv: PNDArrayValue, cb: EmitCodeBuilder): Value[Boolean] = {
    implicit val line = cb.lineNumber
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
    implicit val line = cb.lineNumber
    val shape = pndv.shapes(cb)
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

    pndv.pt.construct(shapeBuilder, stridesBuilder, srvb.end(), cb.emb, region)
  }
}
