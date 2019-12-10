package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
import is.hail.utils._
import scala.reflect.{ClassTag, _}
object PArray {
  def apply(elementType: PType, required: Boolean = false) = new PCanonicalArray(elementType, required)
}

abstract class PArray extends PContainer with PStreamable {
  lazy val virtualType: TArray = TArray(elementType.virtualType, required)

  def copy(required: Boolean): PArray

  def copy(elementType: PType): PArray

  def _asIdent = s"array_of_${elementType.asIdent}"
  def _toPretty = s"Array[$elementType]"

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