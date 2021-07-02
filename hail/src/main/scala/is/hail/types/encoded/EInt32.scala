package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SCanonicalCallCode}
import is.hail.types.physical.stypes.interfaces.SCallValue
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.virtual._
import is.hail.utils._

case object EInt32Optional extends EInt32(false)

case object EInt32Required extends EInt32(true)

class EInt32(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: PValue, out: Value[OutputBuffer]): Unit = {
    val x = v.st match {
      case t: SCanonicalCall => v.asInstanceOf[SCallValue].canonicalCall(cb)
      case t: SInt32 => v.asInt32.intCode(cb)
    }
    cb += out.writeInt(x)
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): PCode = {
    val x = in.readInt()
    t match {
      case TCall => new SCanonicalCallCode(required, x)
      case TInt32 => new SInt32Code(required, x)
    }
  }

  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Int] = in.readInt()

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = cb += in.skipInt()


  def _decodedSType(requestedType: Type): SType = requestedType match {
    case TCall => SCanonicalCall(required)
    case _ => SInt32(required)
  }

  def _asIdent = "int32"

  def _toPretty = "EInt32"

  def setRequired(newRequired: Boolean): EInt32 = EInt32(newRequired)
}

object EInt32 {
  def apply(required: Boolean = false): EInt32 = if (required) EInt32Required else EInt32Optional
}
