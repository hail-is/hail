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

trait Requiredness {
  def required: Boolean
}

object BaseStruct {
  def getMissingIndexAndCount(req: Array[Boolean]): (Array[Int], Int) = {
    val missingIdx = new Array[Int](req.length)
    var i = 0
    var j = 0
    while (i < req.length) {
      missingIdx(i) = j
      if (!req(i))
        j += 1
      i += 1
    }
    (missingIdx, j)
  }
}
