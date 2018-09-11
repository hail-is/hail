package is.hail.expr.types.physical

import is.hail.utils._

final case class PField(name: String, typ: PType, index: Int) {

  def unify(cf: PField): Boolean =
    name == cf.name &&
      typ.unify(cf.typ) &&
      index == cf.index

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
