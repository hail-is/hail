package is.hail.types.encoded

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EBlockMatrixNDArray(elementType: EType, encodeRowMajor: Boolean = false, override val required: Boolean = false) extends EType {
  type DecodedPType = PCanonicalNDArray

  def setRequired(newRequired: Boolean): EBlockMatrixNDArray = EBlockMatrixNDArray(elementType, newRequired)

  override def decodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[DecodedPType] &&
      pt.asInstanceOf[DecodedPType].nDims == 2 &&
      elementType.decodeCompatible(pt.asInstanceOf[DecodedPType].elementType)
  }

  override def encodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[DecodedPType] &&
      pt.asInstanceOf[DecodedPType].nDims == 2 &&
      elementType.encodeCompatible(pt.asInstanceOf[DecodedPType].elementType)
  }

  def _decodedPType(requestedType: Type): PType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TNDArray].elementType)
    PCanonicalNDArray(elementPType, 2, required)
  }

  def _buildEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer]): Unit = {
    val ndarray = PCode(pt, v).asNDArray.memoize(cb, "ndarray")
    val shapes = ndarray.shapes(cb)
    val r = cb.newLocal[Long]("r", shapes(0))
    val c = cb.newLocal[Long]("c", shapes(1))
    val i = cb.newLocal[Long]("i")
    val j = cb.newLocal[Long]("j")
    val writeElemF = elementType.buildEncoder(ndarray.pt.elementType, cb.emb.ecb)

    cb += out.writeInt(r.toI)
    cb += out.writeInt(c.toI)
    cb += out.writeBoolean(encodeRowMajor)
    if (encodeRowMajor) {
      cb.forLoop(cb.assign(i, 0L), i < r, cb.assign(i, i + 1L), {
        cb.forLoop(cb.assign(j, 0L), j < c, cb.assign(j, j + 1L), {
          cb += writeElemF(ndarray.loadElement(FastIndexedSeq(i, j), cb).asPCode.code, out)
        })
      })
    } else {
      cb.forLoop(cb.assign(j, 0L), j < c, cb.assign(j, j + 1L), {
        cb.forLoop(cb.assign(i, 0L), i < r, cb.assign(i, i + 1L), {
          cb += writeElemF(ndarray.loadElement(FastIndexedSeq(i, j), cb).asPCode.code, out)
        })
      })
    }
  }

  def _buildDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = {
    val t = pt.asInstanceOf[PCanonicalNDArray]
    val readElemF = elementType.buildInplaceDecoder(t.elementType, cb.emb.ecb)

    val nRows = cb.newLocal[Long]("rows", in.readInt().toL)
    val nCols = cb.newLocal[Long]("cols", in.readInt().toL)
    val transpose = cb.newLocal[Boolean]("transpose", in.readBoolean())
    val n = cb.newLocal[Int]("length", nRows.toI * nCols.toI)
    val data = cb.newLocal[Long]("data", t.data.pType.allocate(region, n))
    cb += t.data.pType.stagedInitialize(data, n, setMissing=true)

    val i = cb.newLocal[Int]("i")
    cb.forLoop(cb.assign(i, 0), i < n, cb.assign(i, i + 1),
      cb += readElemF(region, t.data.pType.elementOffset(data, n, i), in))

    val shapeBuilder = t.makeShapeBuilder(FastIndexedSeq(nRows, nCols))
    val stridesBuilder = { srvb: StagedRegionValueBuilder =>
      Code(
        srvb.start(),
        srvb.addLong(transpose.mux(nCols.toL * t.elementType.byteSize, t.elementType.byteSize)),
        srvb.advance(),
        srvb.addLong(transpose.mux(t.elementType.byteSize, nRows * t.elementType.byteSize)),
        srvb.advance())
    }

    t.construct(shapeBuilder, stridesBuilder, data, cb.emb, region)
      .tcode[Long]
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb)

    val len = cb.newLocal[Int]("len", in.readInt() * in.readInt())
    val i = cb.newLocal[Int]("i")
    cb += in.skipBoolean()
    cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), cb += skip(r, in))
  }

  def _asIdent = s"ndarray_of_${elementType.asIdent}"
  def _toPretty = s"ENDArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("ENDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }
}
