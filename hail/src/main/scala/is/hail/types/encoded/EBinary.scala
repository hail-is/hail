package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    val addr = coerce[Long](v)
    val bT = pt.asInstanceOf[PBinary]
    val len = cb.newLocal[Int]("len", bT.loadLength(addr))
    cb += out.writeInt(len)
    cb += out.writeBytes(bT.bytesAddress(addr), len)
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber
  ): Code[_] = {
    val bT = pt.asInstanceOf[PBinary]
    val len = cb.newLocal[Int]("len", in.readInt())
    val barray = cb.newLocal[Long]("barray", bT.allocate(region, len))
    cb += bT.storeLength(barray, len)
    cb += in.readBytes(region, bT.bytesAddress(barray), len)
    barray.load()
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit = {
    cb += in.skipBytes(in.readInt())
  }

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PBinary]

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case TBinary => PCanonicalBinary(required)
    case TString => PCanonicalString(required)
  }

  def _asIdent = "binary"
  def _toPretty = "EBinary"

  def setRequired(newRequired: Boolean): EBinary = EBinary(newRequired)
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}
