package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SFloat64, SFloat64Value}
import is.hail.types.virtual._
import is.hail.utils._

case object EFloat64Optional extends EFloat64(false)

case object EFloat64Required extends EFloat64(true)

class EFloat64(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    cb += out.writeDouble(v.asDouble.value)

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue =
    new SFloat64Value(cb.memoize(in.readDouble()))

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipDouble()

  def _decodedSType(requestedType: Type): SType = SFloat64

  def _asIdent = "float64"

  def _toPretty = "EFloat64"

  def setRequired(newRequired: Boolean): EFloat64 = EFloat64(newRequired)
}

object EFloat64 {
  def apply(required: Boolean = false): EFloat64 =
    if (required) EFloat64Required else EFloat64Optional
}
