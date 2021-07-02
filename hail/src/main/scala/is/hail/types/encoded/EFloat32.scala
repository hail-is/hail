package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat32Code}
import is.hail.types.virtual._
import is.hail.utils._

case object EFloat32Optional extends EFloat32(false)

case object EFloat32Required extends EFloat32(true)

class EFloat32(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: PValue, out: Value[OutputBuffer]): Unit = {
    cb += out.writeFloat(v.asFloat.floatCode(cb))
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): PCode = {
    new SFloat32Code(required, in.readFloat())
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = cb += in.skipFloat()

  def _decodedSType(requestedType: Type): SType = SFloat32(required)

  def _asIdent = "float32"

  def _toPretty = "EFloat32"

  def setRequired(newRequired: Boolean): EFloat32 = EFloat32(newRequired)
}

object EFloat32 {
  def apply(required: Boolean = false): EFloat32 = if (required) EFloat32Required else EFloat32Optional
}
