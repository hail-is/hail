package is.hail.expr.types.virtual

import is.hail.expr.ir.Sym

final case class Field(name: Sym, typ: Type, index: Int) {

  def unify(cf: Field): Boolean =
    name == cf.name &&
      typ.unify(cf.typ) &&
      index == cf.index

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append(name)
      sb.append(":")
    } else {
      sb.append(" " * indent)
      sb.append(name)
      sb.append(": ")
    }
    typ.pretty(sb, indent, compact)
  }
}
