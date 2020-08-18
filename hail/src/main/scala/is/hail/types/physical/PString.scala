package is.hail.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.types.virtual.TString

abstract class PString extends PType {
  lazy val virtualType: TString.type = TString

  override def unsafeOrdering(): UnsafeOrdering = PCanonicalBinary(required).unsafeOrdering()

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(this isOfType other)
    PCanonicalBinary(required).codeOrdering(mb, PCanonicalBinary(other.required))
  }

  protected val binaryFundamentalType: PBinary
  override lazy val fundamentalType: PBinary = binaryFundamentalType

  protected val binaryEncodableType: PBinary
  override lazy val encodableType: PBinary = binaryFundamentalType

  def loadLength(boff: Long): Int

  def loadLength(boff: Code[Long]): Code[Int]

  def loadString(boff: Long): String

  def loadString(boff: Code[Long]): Code[String]

  def allocateAndStoreString(region: Region, str: String): Long

  def allocateAndStoreString(mb: EmitMethodBuilder[_], region: Value[Region], str: Code[String]): Code[Long]
}

abstract class PStringCode extends PCode {
  def pt: PString

  def loadLength(): Code[Int]

  def loadString(): Code[String]

  def asBytes(): PBinaryCode
}
