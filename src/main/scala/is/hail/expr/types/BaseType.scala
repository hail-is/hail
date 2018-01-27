package is.hail.expr.types

abstract class BaseType {
  override final def toString: String = toPrettyString(compact = true)

  def toPrettyString(compact: Boolean): String = {
    val sb = new StringBuilder
    pretty(sb, 0, compact = compact)
    sb.result()
  }

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean)
}
