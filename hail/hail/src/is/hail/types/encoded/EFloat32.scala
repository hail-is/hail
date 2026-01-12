package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.asm4s.implicits.{valueToRichCodeInputBuffer, valueToRichCodeOutputBuffer}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat32Value}
import is.hail.types.virtual._

case object EFloat32Optional extends EFloat32(false)

case object EFloat32Required extends EFloat32(true)

class EFloat32(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    cb += out.writeFloat(v.asFloat.value)

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue =
    new SFloat32Value(cb.memoize(in.readFloat()))

  override def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit =
    cb += in.skipFloat()

  override def _decodedSType(requestedType: Type): SType = SFloat32

  override def _asIdent = "float32"

  override def _toPretty = "EFloat32"

  override def setRequired(newRequired: Boolean): EFloat32 = EFloat32(newRequired)
}

object EFloat32 {
  def apply(required: Boolean = false): EFloat32 =
    if (required) EFloat32Required else EFloat32Optional
}
