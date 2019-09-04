package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.check.Arbitrary._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TString
import is.hail.utils._

import scala.reflect.{ClassTag, _}

case object PStringOptional extends PString(false)
case object PStringRequired extends PString(true)

class PString(override val required: Boolean) extends PType {
  lazy val virtualType: TString = TString(required)

  def _toPretty = "String"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("str")
  }

  override def unsafeOrdering(): UnsafeOrdering = PBinary(required).unsafeOrdering()

  override def byteSize: Long = 8

  override def fundamentalType: PBinary = PBinary(required)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    PBinary(required).codeOrdering(mb, PBinary(other.required))
  }

  override def containsPointers: Boolean = true
}

object PString {
  def apply(required: Boolean = false): PString = if (required) PStringRequired else PStringOptional

  def unapply(t: PString): Option[Boolean] = Option(t.required)

  def loadString(region: Region, boff: Long): String = {
    val length = PBinary.loadLength(region, boff)
    new String(region.loadBytes(PBinary.bytesOffset(boff), length))
  }

  def loadString(boff: Code[Long]): Code[String] = {
    val length = PBinary.loadLength(boff)
    Code.newInstance[String, Array[Byte]](
      Region.loadBytes(PBinary.bytesOffset(boff), length))
  }

  def loadString(region: Code[Region], boff: Code[Long]): Code[String] =
    loadString(boff)

  def loadLength(region: Region, boff: Long): Int =
    PBinary.loadLength(region, boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] =
    PBinary.loadLength(region, boff)
}
