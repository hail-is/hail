package is.hail.types.encoded

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EBlockMatrixNDArray(elementType: EType, override val required: Boolean = false) extends EType {
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

  override def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    assert(pnd.elementType.required)
    val ndarray = coerce[Long](v)
    val i = mb.newLocal[Long]("i")
    val j = mb.newLocal[Long]("j")
    val r = mb.newLocal[Long]("r")
    val c = mb.newLocal[Long]("c")
    val writeElemF = elementType.buildEncoder(pnd.elementType, mb.ecb)

    Code(
      r := pnd.loadShape(ndarray, 0),
      c := pnd.loadShape(ndarray, 1),
      out.writeInt(r.toI),
      out.writeInt(c.toI),
      out.writeBoolean(true),
      Code.forLoop(i := 0L, i < r, i := i + 1L,
        Code.forLoop(j := 0L, j < c, j := j + 1L,
          writeElemF(pnd.loadElementToIRIntermediate(FastIndexedSeq(i, j), ndarray, mb),
            out))))
  }

  override def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = {
    val t = pt.asInstanceOf[PCanonicalNDArray]
    val nRows = mb.newLocal[Int]("rows")
    val nCols = mb.newLocal[Int]("cols")
    val transpose = mb.newLocal[Boolean]("transpose")
    val n = mb.newLocal[Int]("length")
    val data = mb.newLocal[Long]("data")

    val readElemF = elementType.buildInplaceDecoder(t.elementType, mb.ecb)
    val i = mb.newLocal[Int]("i")

    val shapeBuilder = t.makeShapeBuilder(FastIndexedSeq(nRows.toL, nCols.toL))
    val stridesBuilder = { srvb: StagedRegionValueBuilder =>
      Code(
        srvb.start(),
        srvb.addLong(transpose.mux(nCols.toL * t.elementType.byteSize, t.elementType.byteSize)),
        srvb.advance(),
        srvb.addLong(transpose.mux(t.elementType.byteSize, nRows.toL * t.elementType.byteSize)),
        srvb.advance())
    }

    Code(
      nRows := in.readInt(),
      nCols := in.readInt(),
      n := nRows * nCols,
      transpose := in.readBoolean(),
      data := t.data.pType.allocate(region, n),
      t.data.pType.stagedInitialize(data, n, setMissing=true),
      Code.forLoop(i := 0, i < n, i := i + 1,
        readElemF(region, t.data.pType.elementOffset(data, n, i), in)),
      t.construct(shapeBuilder, stridesBuilder, data, mb))
  }

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val skip = elementType.buildSkip(mb)
    Code(
      len := in.readInt(),
      len := len * in.readInt(),
      in.skipBoolean(),
      Code.forLoop(i := 0, i < len, i := i + 1, skip(r, in)))
  }

  def _asIdent = s"ndarray_of_${elementType.asIdent}"
  def _toPretty = s"ENDArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("ENDArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }
}

