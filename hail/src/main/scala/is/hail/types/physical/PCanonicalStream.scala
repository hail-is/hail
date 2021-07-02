package is.hail.types.physical

import is.hail.annotations.UnsafeOrdering
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.types.physical.stypes.interfaces
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamCode}
import is.hail.types.virtual.{TStream, Type}

final case class PCanonicalStream(elementType: PType, required: Boolean = false) extends PStream {
  override def unsafeOrdering(): UnsafeOrdering = throw new NotImplementedError()

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCStream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def defaultValue(mb: EmitMethodBuilder[_]): SStreamCode = throw new UnsupportedOperationException

  override def deepRename(t: Type) = deepRenameStream(t.asInstanceOf[TStream])

  private def deepRenameStream(t: TStream): PStream =
    copy(elementType = elementType.deepRename(t.elementType))

  def setRequired(required: Boolean): PCanonicalStream = if (required == this.required) this else this.copy(required = required)

  override def sType: SStream = interfaces.SStream(elementType.sType, required)

  def loadFromNested(addr: Code[Long]): Code[Long] = throw new NotImplementedError()

  override def unstagedLoadFromNested(addr: Long): Long = throw new NotImplementedError()
}
