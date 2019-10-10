package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EFloat32Optional extends EFloat32(false)
case object EFloat32Required extends EFloat32(true)

class EFloat32(override val required: Boolean) extends EType {
  def _buildEncoder(pt: PType, mb: EmitMethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit] = {
    out.writeFloat(coerce[Float](v))
  }

  def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder,
    region: Code[Region],
    in: Code[InputBuffer]
  ): Code[Float] = in.readFloat()

  def _buildSkip(mb: EmitMethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit] = in.skipFloat()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PFloat32]

  def _decodedPType(requestedType: Type): PType = PFloat32(required)

  def _asIdent = "float32"
  def _toPretty = "EFloat32"
}

object EFloat32 {
  def apply(required: Boolean = false): EFloat32 = if (required) EFloat32Required else EFloat32Optional
}
