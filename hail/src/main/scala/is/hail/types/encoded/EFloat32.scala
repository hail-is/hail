package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EFloat32Optional extends EFloat32(false)
case object EFloat32Required extends EFloat32(true)

class EFloat32(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    cb += out.writeFloat(coerce[Float](v))
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber
  ): Code[Float] = in.readFloat()

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit =
    cb += in.skipFloat()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PFloat32]

  def _decodedPType(requestedType: Type): PType = PFloat32(required)

  def _asIdent = "float32"
  def _toPretty = "EFloat32"

  def setRequired(newRequired: Boolean): EFloat32 = EFloat32(newRequired)
}

object EFloat32 {
  def apply(required: Boolean = false): EFloat32 = if (required) EFloat32Required else EFloat32Optional
}
