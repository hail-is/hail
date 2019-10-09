package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EBooleanOptional extends EBoolean(false)
case object EBooleanRequired extends EBoolean(true)

class EBoolean(override val required: Boolean) extends EType {
  def _buildEncoder(pt: PType, mb: EmitMethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit] = {
    out.writeBoolean(coerce[Boolean](v))
  }

  def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder,
    region: Code[Region],
    in: Code[InputBuffer]
  ): Code[Boolean] = in.readBoolean()

  def _buildSkip(mb: EmitMethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit] = in.skipBoolean()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PBoolean]

  def _decodedPType(requestedType: Type): PType = PBoolean(required)

  def _asIdent = "bool"
  def _toPretty = "EBoolean"
}

object EBoolean {
  def apply(required: Boolean = false): EBoolean = if (required) EBooleanRequired else EBooleanOptional
}
