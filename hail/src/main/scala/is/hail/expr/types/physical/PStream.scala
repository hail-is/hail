package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TStream

final case class PStream(elementType: PType, override val required: Boolean = false) extends PContainer {
  lazy val virtualType: TStream = TStream(elementType.virtualType, required)

  val elementByteSize: Long = elementType.byteSize

  val contentsAlignment: Long = elementType.alignment

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("stream<")
    elementType.pyString(sb)
    sb.append('>')
  }
  override val fundamentalType: PStream = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _toPretty = s"Stream[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Stream[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering =
    throw new UnsupportedOperationException("Stream comparison is currently undefined.")
}

