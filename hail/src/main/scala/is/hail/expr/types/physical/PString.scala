package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TString

case object PStringOptional extends PCanonicalString(false)
case object PStringRequired extends PCanonicalString(true)

abstract class PString extends PType {
  lazy val virtualType: TString = TString(required)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("str")
  }

  override def unsafeOrdering(): UnsafeOrdering = PBinary(required).unsafeOrdering()

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    PBinary(required).codeOrdering(mb, PBinary(other.required))
  }

  protected val binaryFundamentalType: PBinary
  override lazy val fundamentalType: PBinary = binaryFundamentalType
}

object PString {
  def apply(required: Boolean = false): PString = if (required) PStringRequired else PStringOptional

  def unapply(t: PString): Option[Boolean] = PCanonicalString.unapply(t)

  def loadString(boff: Long): String = PCanonicalString.loadString(boff)

  def loadString(region: Region, boff: Long): String = PCanonicalString.loadString(region, boff)

  def loadString(boff: Code[Long]): Code[String] = PCanonicalString.loadString(boff)

  def loadString(region: Code[Region], boff: Code[Long]): Code[String] = PCanonicalString.loadString(region, boff)

  def loadLength(region: Region, boff: Long): Int = PCanonicalString.loadLength(region, boff)

  def loadLength(region: Code[Region], boff: Code[Long]): Code[Int] = PCanonicalString.loadLength(region, boff)
}
