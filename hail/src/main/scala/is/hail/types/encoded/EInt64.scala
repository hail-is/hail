package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EInt64Optional extends EInt64(false)
case object EInt64Required extends EInt64(true)

class EInt64(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    cb += out.writeLong(coerce[Long](v))
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber
  ): Code[Long] = in.readLong()

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit =
    cb += in.skipLong()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PInt64]

  def _decodedPType(requestedType: Type): PType = PInt64(required)

  def _asIdent = "int64"
  def _toPretty = "EInt64"

  def setRequired(newRequired: Boolean): EInt64 = EInt64(newRequired)
}

object EInt64 {
  def apply(required: Boolean = false): EInt64 = if (required) EInt64Required else EInt64Optional
}
