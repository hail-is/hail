package is.hail.expr.types.physical
import is.hail.annotations.{CodeOrdering, ExtendedOrdering, Region}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TVoid, Type}

case object PVoid extends PType {
  def virtualType: Type = TVoid

  override val required = true

  def _asIdent = "void"
  override def _toPretty = "Void"

  def copyFromType(mb: MethodBuilder, region: Code[Region], sourcePType: PType, sourceOffset: Code[Long], allowDowncast: Boolean = false, forceDeep: Boolean = false): Code[Long] = ???

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = null
}
