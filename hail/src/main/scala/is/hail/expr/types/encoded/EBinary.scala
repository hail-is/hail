package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType {
  def _buildEncoder(pt: PType, mb: MethodBuilder, v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val addr = coerce[Long](v)
    val len = mb.newLocal[Int]("len")
    val bT = pt.asInstanceOf[PBinary]
    Code(
      len := bT.loadLength(addr),
      out.writeInt(len),
      out.writeBytes(bT.bytesOffset(addr), len))
  }

  def _buildDecoder(
    pt: PType,
    mb: MethodBuilder,
    region: Code[Region],
    in: Code[InputBuffer]
  ): Code[_] = {
    val len = mb.newLocal[Int]("len")
    val barray = mb.newLocal[Long]("barray")
    val bT = pt.asInstanceOf[PBinary]
    Code(
      len := in.readInt(),
      barray := bT.allocate(region, len),
      bT.storeLength(barray, len),
      in.readBytes(region, bT.bytesOffset(barray), len),
      barray.load())
  }

  def _buildSkip(mb: MethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    Code(
      len := in.readInt(),
      in.skipBytes(len))
  }

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PBinary]

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case t: TBinary => PBinary(required)
    case t: TString => PString(required)
  }

  def _asIdent = "binary"
  def _toPretty = "EBinary"
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}
