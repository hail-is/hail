package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case object EInt32Optional extends EInt32(false)
case object EInt32Required extends EInt32(true)

class EInt32(override val required: Boolean) extends EFundamentalType {
  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    cb += out.writeInt(coerce[Int](v))
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber
  ): Code[Int] = in.readInt()

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit =
    cb += in.skipInt()

  override def _compatible(pt: PType): Boolean = pt.isInstanceOf[PInt32]

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case TCall => PCanonicalCall(required)
    case _ => PInt32(required)
  }

  def _asIdent = "int32"
  def _toPretty = "EInt32"

  def setRequired(newRequired: Boolean): EInt32 = EInt32(newRequired)
}

object EInt32 {
  def apply(required: Boolean = false): EInt32 = if (required) EInt32Required else EInt32Optional
}
