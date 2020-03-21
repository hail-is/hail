package is.hail.expr.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TString


abstract class PString extends PType {
  lazy val virtualType: TString.type = TString

  override def unsafeOrdering(): UnsafeOrdering = PBinary(required).unsafeOrdering()

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(this isOfType other)
    PBinary(required).codeOrdering(mb, PBinary(other.required))
  }

  protected val binaryFundamentalType: PBinary
  override lazy val fundamentalType: PBinary = binaryFundamentalType

  def bytesOffset(boff: Long): Long

  def bytesOffset(boff: Code[Long]): Code[Long]

  def loadLength(boff: Long): Int

  def loadLength(boff: Code[Long]): Code[Int]

  def loadString(boff: Long): String

  def loadString(boff: Code[Long]): Code[String]

  def allocateAndStoreString(region: Region, str: String): Long

  def allocateAndStoreString(mb: EmitMethodBuilder[_], region: Value[Region], str: Code[String]): Code[Long]
}

object PString {
  def apply(required: Boolean = false): PString = PCanonicalString(required)

  def unapply(t: PString): Option[Boolean] = PCanonicalString.unapply(t)
}