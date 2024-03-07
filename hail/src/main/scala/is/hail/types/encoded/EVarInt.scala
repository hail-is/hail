package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SCanonicalCallValue}
import is.hail.types.physical.stypes.interfaces.SCallValue
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._
import is.hail.utils._

case object EVarIntOptional extends EVarInt(false)

case object EVarIntRequired extends EVarInt(true)

class EVarInt(val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    v match {
      case v: SCallValue => cb += out.writeLEB128Int(v.canonicalCall(cb))
      case v: SInt32Value => cb += out.writeLEB128Int(v.value)
      case v: SInt64Value => cb += out.writeLEB128Long(v.value)
    }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = {
    val x = in.readLEB128()
    t match {
      case TCall => new SCanonicalCallValue(cb.memoize(x.toI))
      case TInt32 => new SInt32Value(cb.memoize(x.toI))
      case TInt64 => new SInt64Value(cb.memoize(x))
    }
  }

  def _decodedSType(requestedType: Type): SType = requestedType match {
    case TCall => SCanonicalCall
    case TInt32 => SInt32
    case TInt64 => SInt64
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += Code.toUnit(in.readLEB128())

  def _asIdent = "varint"

  def _toPretty = "EVarInt"

  def setRequired(newRequired: Boolean): EVarInt = EVarInt(newRequired)
}

object EVarInt {
  def apply(required: Boolean = false): EVarInt = if (required) EVarIntRequired else EVarIntOptional
}
