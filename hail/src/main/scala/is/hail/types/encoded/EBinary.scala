package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue, SString}
import is.hail.types.virtual._
import is.hail.utils._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType {

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {

    def writeCanonicalBinary(bin: SBinaryPointerValue): Unit = {
      val len = bin.loadLength(cb)
      cb += out.writeInt(len)
      cb += out.writeBytes(bin.bytesAddress(), len)
    }

    def writeBytes(bytes: Value[Array[Byte]]): Unit = {
      cb += out.writeInt(bytes.length())
      cb += out.write(bytes)
    }

    v.st match {
      case SBinaryPointer(_) => writeCanonicalBinary(v.asInstanceOf[SBinaryPointerValue])
      case SStringPointer(_) =>
        writeCanonicalBinary(v.asInstanceOf[SStringPointerValue].binaryRepr())
      case _: SBinary => writeBytes(v.asInstanceOf[SBinaryValue].loadBytes(cb))
      case _: SString => writeBytes(v.asString.toBytes(cb).loadBytes(cb))
    }
  }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = {
    val t1 = decodedSType(t)
    val pt = t1 match {
      case SStringPointer(t) => t.binaryRepresentation
      case SBinaryPointer(t) => t
    }

    val bT = pt
    val len = cb.newLocal[Int]("len", in.readInt())
    val barray = cb.newLocal[Long]("barray", bT.allocate(region, len))
    bT.storeLength(cb, barray, len)
    cb += in.readBytes(region, bT.bytesAddress(barray), len)
    t1 match {
      case t: SStringPointer => new SStringPointerValue(t, barray)
      case t: SBinaryPointer => new SBinaryPointerValue(t, barray)
    }
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipBytes(in.readInt())

  def _decodedSType(requestedType: Type): SType = requestedType match {
    case TBinary => SBinaryPointer(PCanonicalBinary(false))
    case TString => SStringPointer(PCanonicalString(false))
  }

  def _asIdent = "binary"
  def _toPretty = "EBinary"

  def setRequired(newRequired: Boolean): EBinary = EBinary(newRequired)
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}
