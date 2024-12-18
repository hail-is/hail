package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SInt64, SInt64Value}
import is.hail.types.virtual._
import is.hail.utils._

case object EInt64Optional extends EInt64(false)

case object EInt64Required extends EInt64(true)

class EInt64(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    cb += out.writeLong(v.asLong.value)

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue =
    new SInt64Value(cb.memoize(in.readLong()))

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipLong()

  def _decodedSType(requestedType: Type): SType = SInt64

  def _asIdent = "int64"

  def _toPretty = "EInt64"

  def setRequired(newRequired: Boolean): EInt64 = EInt64(newRequired)
}

object EInt64 {
  def apply(required: Boolean = false): EInt64 = if (required) EInt64Required else EInt64Optional
}
