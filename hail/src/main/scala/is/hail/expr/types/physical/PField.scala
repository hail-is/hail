package is.hail.expr.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.Code
import is.hail.utils._

final class StaticallyKnownField[T, U](
  val pType: T,
  val load: (Code[Region], Code[Long]) => Code[U]
)

final case class PField(name: String, typ: PType, index: Int) {
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append(prettyIdentifier(name))
      sb.append(":")
    } else {
      sb.append(" " * indent)
      sb.append(prettyIdentifier(name))
      sb.append(": ")
    }
    typ.pretty(sb, indent, compact)
  }
}
