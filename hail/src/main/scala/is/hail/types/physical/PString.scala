package is.hail.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.expr.ir.{EmitMethodBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.interfaces.{SStringCode, SStringValue}
import is.hail.types.virtual.TString

abstract class PString extends PType {
  lazy val virtualType: TString.type = TString

  override def unsafeOrdering(): UnsafeOrdering = PCanonicalBinary(required).unsafeOrdering()

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(this isOfType other)
    new CodeOrderingCompareConsistentWithOthers {
      val ord = PCanonicalBinary(required).codeOrdering(mb, PCanonicalBinary(other.required))
      def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        ord.compareNonnull(cb, x.asString.asBytes(), y.asString.asBytes())
    }
  }

  val binaryRepresentation: PBinary
  override lazy val fundamentalType: PBinary = binaryRepresentation

  protected val binaryEncodableType: PBinary
  override lazy val encodableType: PBinary = binaryRepresentation

  def loadLength(boff: Long): Int

  def loadLength(boff: Code[Long]): Code[Int]

  def loadString(boff: Long): String

  def loadString(boff: Code[Long]): Code[String]

  def allocateAndStoreString(region: Region, str: String): Long

  def allocateAndStoreString(mb: EmitMethodBuilder[_], region: Value[Region], str: Code[String]): Code[Long]
}

abstract class PStringCode extends PCode with SStringCode {
  def pt: PString

  def asBytes(): PBinaryCode
}

abstract class PStringValue extends PValue with SStringValue {
  def pt: PString

  def get: PStringCode
}
