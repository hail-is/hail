package is.hail.expr.types.physical

import is.hail.annotations.CodeOrdering
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
object PArray {
  def apply(elementType: PType, required: Boolean = false) = new PCanonicalArray(elementType, required)
}

abstract class PArray extends PContainer with PStreamable {
  lazy val virtualType: TArray = TArray(elementType.virtualType, required)

  def copy(required: Boolean): PArray

  def copy(elementType: PType): PArray

  def _asIdent = s"array_of_${elementType.asIdent}"
  def _toPretty = s"Array[$elementType]"

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    CodeOrdering.iterableOrdering(this, other.asInstanceOf[PArray], mb)
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("array<")
    elementType.pyString(sb)
    sb.append('>')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def containsPointers: Boolean = true
}