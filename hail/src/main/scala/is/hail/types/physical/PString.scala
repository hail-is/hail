package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.virtual.TString

abstract class PString extends PType {
  lazy val virtualType: TString.type = TString

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering =
    PCanonicalBinary(required).unsafeOrdering(sm)

  val binaryRepresentation: PBinary

  def loadLength(boff: Long): Int

  def loadLength(boff: Code[Long]): Code[Int]

  def loadString(boff: Long): String

  def loadString(boff: Code[Long]): Code[String]

  def allocateAndStoreString(region: Region, str: String): Long

  def allocateAndStoreString(cb: EmitCodeBuilder, region: Value[Region], str: Code[String])
    : Value[Long]
}
