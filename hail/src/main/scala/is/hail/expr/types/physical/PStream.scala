package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.{TStream, Type}

final case class PStream(elementType: PType, required: Boolean = false) extends PIterable {
  lazy val virtualType: TStream = TStream(elementType.virtualType)

  override val fundamentalType: PStream = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _asIdent = s"stream_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PStream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering =
    throw new UnsupportedOperationException("PStream comparison is currently undefined.")

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean) =
    throw new UnsupportedOperationException("PStream copyFromType is currently undefined")

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean) =
    throw new UnsupportedOperationException("PStream copyFromType is currently undefined")

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    throw new UnsupportedOperationException("PStream copyFromTypeAndStackValue is currently undefined")

  override def deepRename(t: Type) = deepRenameStream(t.asInstanceOf[TStream])

  private def deepRenameStream(t: TStream): PStream =
    PStream(this.elementType.deepRename(t.elementType), this.required)

  def constructAtAddress(mb: MethodBuilder, addr: Code[Long], region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Unit] =
    throw new NotImplementedError(s"$this is not constructable")

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Unit =
    throw new NotImplementedError(s"$this is not constructable")

  def setRequired(required: Boolean) = if(required == this.required) this else this.copy(required = required)
}

