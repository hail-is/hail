package is.hail.expr.types

abstract class BaseType {
  override final def toString: String = {
    val sb = new StringBuilder
    pyString(sb)
    sb.result()
  }

  def toPrettyString(indent: Int, compact: Boolean): String = {
    val sb = new StringBuilder
    pretty(sb, indent, compact = compact)
    sb.result()
  }

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean)

  def parsableString(): String = toPrettyString(0, compact = true)

  def pyString(sb: StringBuilder) = pretty(sb, 0, compact = true)
}
