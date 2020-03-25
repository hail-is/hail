package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, MethodBuilder, Value}
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

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    throw new UnsupportedOperationException("PStream comparison is currently undefined.")

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean) =
    throw new UnsupportedOperationException("PStream copyFromType is currently undefined")

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean) =
    throw new UnsupportedOperationException("PStream copyFromType is currently undefined")

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    throw new UnsupportedOperationException("PStream copyFromTypeAndStackValue is currently undefined")

  override def deepRename(t: Type) = deepRenameStream(t.asInstanceOf[TStream])

  private def deepRenameStream(t: TStream): PStream =
    PStream(this.elementType.deepRename(t.elementType), this.required)

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    throw new NotImplementedError(s"$this is not constructable")

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    throw new NotImplementedError(s"$this is not constructable")

  def setRequired(required: Boolean) = if(required == this.required) this else this.copy(required = required)
}

