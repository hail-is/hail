package is.hail.expr.types

abstract class BaseType {
  override final def toString: String = toPrettyString(0, compact = true)

  def toPrettyString(indent: Int, compact: Boolean): String = {
    val sb = new StringBuilder
    pretty(sb, indent, compact = compact)
    sb.result()
  }

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean)
}
