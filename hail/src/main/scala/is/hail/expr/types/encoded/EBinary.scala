package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType {
  def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val addr = coerce[Long](v)
    val len = mb.newLocal[Int]("len")
    val bT = pt.asInstanceOf[PBinary]
    Code(
      len := bT.loadLength(addr),
      out.writeInt(len),
      out.writeBytes(bT.bytesAddress(addr), len))
  }

  def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[_] = {
    val len = mb.newLocal[Int]("len")
    val barray = mb.newLocal[Long]("barray")
    val bT = pt.asInstanceOf[PBinary]
    Code(
      len := in.readInt(),
      barray := bT.allocate(region, len),
      bT.storeLength(barray, len),
      in.readBytes(region, bT.bytesAddress(barray), len),
      barray.load())
  }

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    Code(
      len := in.readInt(),
      in.skipBytes(len))
  }

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PBinary]

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case TBinary => PBinary(required)
    case TString => PString(required)
  }

  def _asIdent = "binary"
  def _toPretty = "EBinary"
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}
