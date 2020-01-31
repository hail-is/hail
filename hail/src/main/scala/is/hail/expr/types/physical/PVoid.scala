package is.hail.expr.types.physical
import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TVoid, Type}

case object PVoid extends PType {
  def virtualType: Type = TVoid

  override val required = true

  def _asIdent = "void"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PVoid")

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = null

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean) =
    throw new UnsupportedOperationException("PVoid copyFromType is currently undefined")

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean) =
    throw new UnsupportedOperationException("PVoid copyFromType is currently undefined")

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    throw new UnsupportedOperationException("PVoid copyFromTypeAndStackValue is currently undefined")

  def storeShallowAtOffset(dstAddress: Code[Long], srcAddress: Code[Long]): Code[Unit] =
    throw new UnsupportedOperationException("PVoid storeShallowAtOffset is currently undefined")

  def storeShallowAtOffset(dstAddress: Long, srcAddress: Long) =
    throw new UnsupportedOperationException("PVoid storeShallowAtOffset is currently undefined")
}
