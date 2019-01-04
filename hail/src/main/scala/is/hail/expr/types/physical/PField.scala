package is.hail.expr.types.physical

import is.hail.expr.ir.Sym

final case class PField(name: Sym, typ: PType, index: Int) {
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
