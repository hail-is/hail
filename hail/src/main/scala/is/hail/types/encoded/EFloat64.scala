package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EFloat64Optional extends EFloat64(false)
case object EFloat64Required extends EFloat64(true)

class EFloat64(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    cb += out.writeDouble(coerce[Double](v))
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber
  ): Code[Double] = in.readDouble()

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit =
    cb += in.skipDouble()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PFloat64]

  def _decodedPType(requestedType: Type): PType = PFloat64(required)

  def _asIdent = "float64"
  def _toPretty = "EFloat64"

  def setRequired(newRequired: Boolean): EFloat64 = EFloat64(newRequired)
}

object EFloat64 {
  def apply(required: Boolean = false): EFloat64 = if (required) EFloat64Required else EFloat64Optional
}
