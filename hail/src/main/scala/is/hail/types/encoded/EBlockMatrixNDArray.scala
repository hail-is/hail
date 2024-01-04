package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.SNDArrayPointer
import is.hail.types.physical.stypes.interfaces.SNDArrayValue
import is.hail.types.virtual._
import is.hail.utils._

final case class EBlockMatrixNDArray(
  elementType: EType,
  encodeRowMajor: Boolean = false,
  override val required: Boolean = false,
) extends EType {
  type DecodedPType = PCanonicalNDArray

  def setRequired(newRequired: Boolean): EBlockMatrixNDArray =
    EBlockMatrixNDArray(elementType, newRequired)

  def _decodedSType(requestedType: Type): SType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TNDArray].elementType)
    SNDArrayPointer(PCanonicalNDArray(elementPType, 2, false))
  }

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val ndarray = v.asInstanceOf[SNDArrayValue]
    val shapes = ndarray.shapes
    val r = cb.newLocal[Long]("r", shapes(0))
    val c = cb.newLocal[Long]("c", shapes(1))
    val i = cb.newLocal[Long]("i")
    val j = cb.newLocal[Long]("j")
    val writeElemF = elementType.buildEncoder(ndarray.st.elementType, cb.emb.ecb)

    cb += out.writeInt(r.toI)
    cb += out.writeInt(c.toI)
    cb += out.writeBoolean(encodeRowMajor)
    if (encodeRowMajor) {
      cb.for_(
        cb.assign(i, 0L),
        i < r,
        cb.assign(i, i + 1L),
        cb.for_(
          cb.assign(j, 0L),
          j < c,
          cb.assign(j, j + 1L),
          writeElemF(cb, ndarray.loadElement(FastSeq(i, j), cb), out),
        ),
      )
    } else {
      cb.for_(
        cb.assign(j, 0L),
        j < c,
        cb.assign(j, j + 1L),
        cb.for_(
          cb.assign(i, 0L),
          i < r,
          cb.assign(i, i + 1L),
          writeElemF(cb, ndarray.loadElement(FastSeq(i, j), cb), out),
        ),
      )
    }
  }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = {
    val st = decodedSType(t).asInstanceOf[SNDArrayPointer]
    val pt = st.pType
    val readElemF = elementType.buildInplaceDecoder(pt.elementType, cb.emb.ecb)

    val nRows = cb.newLocal[Long]("rows", in.readInt().toL)
    val nCols = cb.newLocal[Long]("cols", in.readInt().toL)
    val transpose = cb.newLocal[Boolean]("transpose", in.readBoolean())

    val stride0 = cb.newLocal[Long](
      "stride0",
      transpose.mux(nCols.toL * pt.elementType.byteSize, pt.elementType.byteSize),
    )
    val stride1 = cb.newLocal[Long](
      "stride1",
      transpose.mux(pt.elementType.byteSize, nRows * pt.elementType.byteSize),
    )

    val n = cb.newLocal[Int]("length", nRows.toI * nCols.toI)

    val (tFirstElementAddress, tFinisher) =
      pt.constructDataFunction(IndexedSeq(nRows, nCols), IndexedSeq(stride0, stride1), cb, region)
    val currElementAddress =
      cb.newLocal[Long]("eblockmatrix_ndarray_currElementAddress", tFirstElementAddress)

    val i = cb.newLocal[Int]("i")
    cb.for_(
      cb.assign(i, 0),
      i < n,
      cb.assign(i, i + 1), {
        readElemF(cb, region, currElementAddress, in)
        cb.assign(currElementAddress, currElementAddress + pt.elementType.byteSize)
      },
    )

    tFinisher(cb)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb.ecb)

    val len = cb.newLocal[Int]("len", in.readInt() * in.readInt())
    val i = cb.newLocal[Int]("i")
    cb += in.skipBoolean()
    cb.for_(cb.assign(i, 0), i < len, cb.assign(i, i + 1), skip(cb, r, in))
  }

  override def _asIdent: String =
    s"bm_ndarray_${if (encodeRowMajor) "row" else "column"}_major_of_${elementType.asIdent}"

  def _toPretty = s"ENDArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("ENDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }
}
