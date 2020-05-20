package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EBooleanOptional extends EBoolean(false)
case object EBooleanRequired extends EBoolean(true)

class EBoolean(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    out.writeBoolean(coerce[Boolean](v))
  }

  def _buildFundamentalDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Boolean] = in.readBoolean()

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = in.skipBoolean()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PBoolean]

  def _decodedPType(requestedType: Type): PType = PBoolean(required)

  def _asIdent = "bool"
  def _toPretty = "EBoolean"

  def setRequired(newRequired: Boolean): EBoolean = EBoolean(newRequired)
}

object EBoolean {
  def apply(required: Boolean = false): EBoolean = if (required) EBooleanRequired else EBooleanOptional
}
