package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete.{SBinaryPointer, SBinaryPointerCode, SBinaryPointerSettable, SStringPointer, SStringPointerCode, SStringPointerSettable}
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue, SString, SStringValue}
import is.hail.utils._

case object EBinaryOptional extends EBinary(false)
case object EBinaryRequired extends EBinary(true)

class EBinary(override val required: Boolean) extends EType {

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {

    def writeCanonicalBinary(bin: SBinaryPointerSettable): Unit = {
      val len = cb.newLocal[Int]("len", bin.loadLength())
      cb += out.writeInt(len)
      cb += out.writeBytes(bin.bytesAddress(), len)
    }

    def writeBytes(_bytes: Code[Array[Byte]]): Unit = {
      val bytes = cb.newLocal[Array[Byte]]("bytes", _bytes)
      val len = cb.newLocal[Int]("len", bytes.length())
      cb += out.writeInt(bytes.length())
      cb += out.write(bytes)
    }

    v.st match {
      case SBinaryPointer(_) => writeCanonicalBinary(v.asInstanceOf[SBinaryPointerSettable])
      case SStringPointer(_) => writeCanonicalBinary(v.asInstanceOf[SStringPointerSettable].binaryRepr())
      case _: SBinary => writeBytes(v.asInstanceOf[SBinaryValue].loadBytes())
      case _: SString => writeBytes(v.asString.toBytes().loadBytes())
    }
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = {
    val t1 = decodedSType(t)
    val pt = t1 match {
      case SStringPointer(t) => t.binaryRepresentation
      case SBinaryPointer(t) => t
    }

    val bT = pt.asInstanceOf[PBinary]
    val len = cb.newLocal[Int]("len", in.readInt())
    val barray = cb.newLocal[Long]("barray", bT.allocate(region, len))
    cb += bT.storeLength(barray, len)
    cb += in.readBytes(region, bT.bytesAddress(barray), len)
    t1 match {
      case t: SStringPointer => new SStringPointerCode(t, barray)
      case t: SBinaryPointer => new SBinaryPointerCode(t, barray)
    }
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    cb += in.skipBytes(in.readInt())
  }

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
