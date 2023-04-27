package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.PCanonicalNDArray
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.SNDArrayPointer
import is.hail.types.physical.stypes.interfaces.SNDArrayValue
import is.hail.types.physical.stypes.primitives.SFloat64
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils.FastIndexedSeq

// FIXME numpy format should not be a hail native serialized format, move this to ValueReader/Writer
final case class ENumpyBinaryNDArray(nRows: Long, nCols: Long, required: Boolean) extends EType {
  type DecodedPType = PCanonicalNDArray
  val elementType = EFloat64(true)

  def setRequired(newRequired: Boolean): ENumpyBinaryNDArray = ENumpyBinaryNDArray(nRows, nCols, newRequired)

  def _decodedSType(requestedType: Type): SType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TNDArray].elementType)
    SNDArrayPointer(PCanonicalNDArray(elementPType, 2, false))
  }

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val ndarray = v.asInstanceOf[SNDArrayValue]
    assert(ndarray.st.elementType == SFloat64)
    val i = cb.newLocal[Long]("i")
    val j = cb.newLocal[Long]("j")
    val writeElemF = elementType.buildEncoder(ndarray.st.elementType, cb.emb.ecb)

    cb.forLoop(cb.assign(i, 0L), i < nRows, cb.assign(i, i + 1L), {
      cb.forLoop(cb.assign(j, 0L), j < nCols, cb.assign(j, j + 1L), {
        writeElemF(cb, ndarray.loadElement(FastIndexedSeq(i, j), cb), out)
      })
    })

  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SValue = {
    val st = decodedSType(t).asInstanceOf[SNDArrayPointer]
    val pt = st.pType
    val readElemF = elementType.buildInplaceDecoder(pt.elementType, cb.emb.ecb)

    val stride0 = cb.newLocal[Long]("stride0", nCols * pt.elementType.byteSize)
    val stride1 = cb.newLocal[Long]("stride1", pt.elementType.byteSize)

    val n = cb.newLocal[Long]("length", nRows * nCols)

    val (tFirstElementAddress, tFinisher) = pt.constructDataFunction(IndexedSeq(nRows, nCols), IndexedSeq(stride0, stride1), cb, region)
    val currElementAddress = cb.newLocal[Long]("eblockmatrix_ndarray_currElementAddress", tFirstElementAddress)

    val i = cb.newLocal[Long]("i")
    cb.forLoop(cb.assign(i, 0L), i < n, cb.assign(i, i + 1L), {
      readElemF(cb, region, currElementAddress, in)
      cb.assign(currElementAddress, currElementAddress + pt.elementType.byteSize)
    })

    tFinisher(cb)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    ???
  }

  def _asIdent = s"ndarray_of_${ elementType.asIdent }"

  def _toPretty = s"ENDArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("ENDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

}
