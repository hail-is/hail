package is.hail.expr.types

import is.hail.utils.HailException

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

// The type of a recur node is an exception. We catch them in inferring the type of If nodes in
// order to type them properly.
final class RecurType(
  override val msg: String,
  override val logMsg: Option[String] = None,
  cause: Throwable = null) extends HailException(msg, logMsg, cause)
