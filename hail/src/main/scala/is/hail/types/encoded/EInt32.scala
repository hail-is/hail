package is.hail.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete.{SCanonicalCall, SCanonicalCallCode}
import is.hail.types.physical.stypes.interfaces.{SCall, SCallValue}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.virtual._
import is.hail.utils._
import org.json4s.JsonAST.JString
import org.json4s.{JBool, JObject, JValue}

case object EInt32Optional extends EInt32(false)

case object EInt32Required extends EInt32(true)

class EInt32(override val required: Boolean) extends EType {
  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val x = v.st match {
      case _: SCall => v.asInstanceOf[SCallValue].canonicalCall(cb)
      case SInt32 => v.asInt32.intCode(cb)
    }
    cb += out.writeInt(x)
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = {
    val x = in.readInt()
    t match {
      case TCall => new SCanonicalCallCode(x)
      case TInt32 => new SInt32Code(x)
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
    case TCall => SCanonicalCall
    case _ => SInt32
  }

  def _asIdent = "int32"

  def _toPretty = "EInt32"

  def setRequired(newRequired: Boolean): EInt32 = EInt32(newRequired)

  override def jsonRepresentation: JValue = JObject(("name", JString("EInt32")), ("required", JBool(this.required)))
}

object EInt32 {
  def apply(required: Boolean = false): EInt32 = if (required) EInt32Required else EInt32Optional
}
