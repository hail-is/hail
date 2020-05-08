package is.hail.types.physical

import is.hail.asm4s._
import is.hail.annotations.CodeOrdering
import is.hail.annotations.{UnsafeOrdering, _}
import is.hail.expr.ir.{ConsistentEmitCodeOrdering, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitModuleBuilder}
import is.hail.types.virtual.TString

abstract class PString extends PType {
  lazy val virtualType: TString.type = TString

  override def unsafeOrdering(): UnsafeOrdering = PCanonicalBinary(required).unsafeOrdering()

  override def codeOrdering2(modb: EmitModuleBuilder, other: PType): ConsistentEmitCodeOrdering = {
    val otherBinary = other.asInstanceOf[PString].binaryFundamentalType
    val binord = modb.getCodeOrdering2(binaryFundamentalType, otherBinary)
    new ConsistentEmitCodeOrdering(modb, this, other) {
      def emitCompare(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Int] = {
        val lhs = EmitCode.present(lhsc.asString.asBytes())
        val rhs = EmitCode.present(rhsc.asString.asBytes())

        binord.compare(cb, lhs, rhs)
      }

      override def emitEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
        val lhs = EmitCode.present(lhsc.asString.asBytes())
        val rhs = EmitCode.present(rhsc.asString.asBytes())

        binord.equiv(cb, lhs, rhs)
      }
    }
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(this isOfType other)
    PCanonicalBinary(required).codeOrdering(mb, PCanonicalBinary(other.required))
  }

  protected val binaryFundamentalType: PBinary
  override lazy val fundamentalType: PBinary = binaryFundamentalType

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
