package is.hail.types.virtual

import is.hail.utils._

final case class Field(name: String, typ: Type, index: Int) {

  def unify(cf: Field): Boolean =
    name == cf.name &&
      typ.unify(cf.typ) &&
      index == cf.index

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
