package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeInputBuffer, valueToRichCodeOutputBuffer}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SCanonicalCallValue}
import is.hail.types.physical.stypes.interfaces.{SCall, SCallValue}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value, SInt64, SInt64Value}
import is.hail.types.virtual._

case object EVarintOptional extends EVarint(false)
case object EVarintRequired extends EVarint(true)

class EVarint(override val required: Boolean) extends EIntegral {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    v.st match {
      case _: SCall => cb += out.writeVarint(v.asInstanceOf[SCallValue].canonicalCall(cb))
      case SInt32 => cb += out.writeVarint(v.asInt32.value)
      case SInt64 => cb += out.writeVarintLong(v.asInt64.value)
    }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = t match {
    case TCall => new SCanonicalCallValue(cb.memoize(in.readVarint()))
    case TInt32 => new SInt32Value(cb.memoize(in.readVarint()))
    case TInt64 => new SInt64Value(cb.memoize(in.readVarintLong()))
  }

  override def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipVarint()

  override def _decodedSType(requestedType: Type): SType = requestedType match {
    case TCall => SCanonicalCall
    case TInt32 => SInt32
    case TInt64 => SInt64
  }

  override def _asIdent = "varint"

  override def _toPretty = "EVarint"

  override def setRequired(newRequired: Boolean): EVarint = EVarint(newRequired)
}

object EVarint {
  def apply(required: Boolean = false): EVarint = if (required) EVarintRequired else EVarintOptional
}
