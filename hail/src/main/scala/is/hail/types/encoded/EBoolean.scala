package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SBoolean, SBooleanValue}
import is.hail.types.virtual._
import is.hail.utils._

case object EBooleanOptional extends EBoolean(false)

case object EBooleanRequired extends EBoolean(true)

class EBoolean(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    cb += out.writeBoolean(v.asBoolean.value)

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue =
    new SBooleanValue(cb.memoize(in.readBoolean()))

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipBoolean()

  def _decodedSType(requestedType: Type): SType = SBoolean

  def _asIdent = "bool"

  def _toPretty = "EBoolean"

  def setRequired(newRequired: Boolean): EBoolean = EBoolean(newRequired)
}

object EBoolean {
  def apply(required: Boolean = false): EBoolean =
    if (required) EBooleanRequired else EBooleanOptional
}
