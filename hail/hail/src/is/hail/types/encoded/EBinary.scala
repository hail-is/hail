package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeInputBuffer, valueToRichCodeOutputBuffer}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue, SString}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.virtual._

case object EBinaryLegacyFullWidthIntegerLengthOptional extends EBinary(false, EInt32Required)
case object EBinaryLegacyFullWidthIntegerLengthRequired extends EBinary(true, EInt32Required)

case object EBinaryOptional extends EBinary(false, EVarintRequired)
case object EBinaryRequired extends EBinary(true, EVarintRequired)

class EBinary(override val required: Boolean, lengthEType: EIntegral) extends EType {
  def writeLength(cb: EmitCodeBuilder, out: Value[OutputBuffer], len: Code[Int]): Unit = {
    val lsv = new SInt32Value(cb.memoize(len))
    lengthEType.buildEncoder(SInt32, cb.emb.ecb).apply(cb, lsv, out)
  }

  def readLength(cb: EmitCodeBuilder, region: Value[Region], in: Value[InputBuffer]): Value[Int] =
    lengthEType.buildDecoder(TInt32, cb.emb.ecb).apply(cb, region, in).asInt32.value

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {

    def writeCanonicalBinary(bin: SBinaryPointerValue): Unit = {
      val len = bin.loadLength(cb)
      writeLength(cb, out, len)
      cb += out.writeBytes(bin.bytesAddress(), len)
    }

    def writeBytes(bytes: Value[Array[Byte]]): Unit = {
      writeLength(cb, out, bytes.length())
      cb += out.write(bytes)
    }

    v.st match {
      case SBinaryPointer(_) => writeCanonicalBinary(v.asInstanceOf[SBinaryPointerValue])
      case SStringPointer(_) =>
        writeCanonicalBinary(v.asInstanceOf[SStringPointerValue].binaryRepr)
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
    val len = readLength(cb, region, in)
    val barray = cb.newLocal[Long]("barray", bT.allocate(region, len))
    bT.storeLength(cb, barray, len)
    cb += in.readBytes(region, bT.bytesAddress(barray), len)
    t1 match {
      case t: SStringPointer => new SStringPointerValue(t, barray)
      case t: SBinaryPointer => new SBinaryPointerValue(t, barray)
    }
  }

  override def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipBytes(readLength(cb, r, in))

  override def _decodedSType(requestedType: Type): SType = requestedType match {
    case TBinary => SBinaryPointer(PCanonicalBinary(false))
    case TString => SStringPointer(PCanonicalString(false))
  }

  // It was a mistake to simply use write/readInt for lengths in 'version 1' so
  // the new serialization gets to be version 2 while the old version has
  // a very long and descriptive name to indicate what it is and to discourage
  // use
  private def ver: String = lengthEType match {
    case EInt32Required => "LegacyFullWidthIntegerLength"
    case EVarintRequired => "2"
  }

  override def _asIdent = s"binary"
  override def _toPretty = s"EBinary$ver"

  override def setRequired(newRequired: Boolean): EBinary = lengthEType match {
    case EInt32Required => EBinaryLegacyFullWidthIntegerLength(newRequired)
    case EVarintRequired => EBinary(newRequired)
  }
}

object EBinary {
  def apply(required: Boolean = false): EBinary = if (required) EBinaryRequired else EBinaryOptional
}

object EBinaryLegacyFullWidthIntegerLength {
  def apply(required: Boolean = false): EBinary =
    if (required) EBinaryLegacyFullWidthIntegerLengthRequired
    else EBinaryLegacyFullWidthIntegerLengthOptional
}
