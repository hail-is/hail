package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EInt64Optional extends EInt64(false)
case object EInt64Required extends EInt64(true)

class EInt64(override val required: Boolean) extends EType {
  def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    out.writeLong(coerce[Long](v))
  }

  def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = in.readLong()

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = in.skipLong()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PInt64]

  def _decodedPType(requestedType: Type): PType = PInt64(required)

  def _asIdent = "int64"
  def _toPretty = "EInt64"
}

object EInt64 {
  def apply(required: Boolean = false): EInt64 = if (required) EInt64Required else EInt64Optional
}
