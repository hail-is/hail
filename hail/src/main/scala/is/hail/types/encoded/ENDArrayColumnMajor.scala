package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete.SNDArrayPointer
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayValue}
import is.hail.types.physical.PCanonicalNDArray
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils._

case class ENDArrayColumnMajor(elementType: EType, nDims: Int, required: Boolean = false) extends EContainer {

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val ndarray = v.asInstanceOf[SNDArrayValue]

    val shapes = ndarray.shapes
    shapes.foreach(s => cb += out.writeLong(s))

    SNDArray.coiterate(cb, (ndarray, "A")){
      case Seq(elt) =>
        elementType.buildEncoder(elt.st, cb.emb.ecb)
          .apply(cb, elt, out)
    }
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SValue = {
    val st = decodedSType(t).asInstanceOf[SNDArrayPointer]
    val pnd = st.pType
    val readElemF = elementType.buildInplaceDecoder(pnd.elementType, cb.emb.ecb)

    val shapeVars = (0 until nDims).map(i => cb.newLocal[Long](s"ndarray_decoder_shape_$i", in.readLong()))
    val totalNumElements = cb.newLocal[Long]("ndarray_decoder_total_num_elements", 1L)
    shapeVars.foreach { s =>
      cb.assign(totalNumElements, totalNumElements * s)
    }
    val strides = pnd.makeColumnMajorStrides(shapeVars, cb)

    val (pndFirstElementAddress, pndFinisher) = pnd.constructDataFunction(shapeVars, strides, cb, region)

    val currElementAddress = cb.newLocal[Long]("eblockmatrix_ndarray_currElementAddress", pndFirstElementAddress)

    val dataIdx = cb.newLocal[Int]("ndarray_decoder_data_idx")
    cb.for_(cb.assign(dataIdx, 0), dataIdx < totalNumElements.toI, cb.assign(dataIdx, dataIdx + 1), {
      readElemF(cb, region, currElementAddress, in)
      cb.assign(currElementAddress, currElementAddress + pnd.elementType.byteSize)
    })

    pndFinisher(cb)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb.ecb)

    val numElements = cb.newLocal[Long]("ndarray_skipper_total_num_elements",
      (0 until nDims).foldLeft(const(1L).get) { (p, i) => p * in.readLong() })
    val i = cb.newLocal[Long]("ndarray_skipper_data_idx")
    cb.for_(cb.assign(i, 0L), i < numElements, cb.assign(i, i + 1L), skip(cb, r, in))
  }

  def _decodedSType(requestedType: Type): SType = {
    val requestedTNDArray = requestedType.asInstanceOf[TNDArray]
    val elementPType = elementType.decodedPType(requestedTNDArray.elementType)
    SNDArrayPointer(PCanonicalNDArray(elementPType, requestedTNDArray.nDims, false))
  }

  override def setRequired(required: Boolean): EType = ENDArrayColumnMajor(elementType, nDims, required)

  override def _asIdent = s"ndarray_of_${ elementType.asIdent }"

  override def _toPretty = s"ENDArrayColumnMajor[$elementType,$nDims]"
}
