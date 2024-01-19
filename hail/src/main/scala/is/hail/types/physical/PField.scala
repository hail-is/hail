package is.hail.types.physical

import is.hail.utils._

final case class PField(name: String, typ: PType, index: Int) {
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
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
