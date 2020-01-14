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
  def getSetMissingness(req: Array[Boolean], missingIdx: Array[Int]): Int = {
    assert(missingIdx.length == req.length)
    var i = 0
    var j = 0
    while (i < req.length) {
      val r = req(i)
      missingIdx(i) = j
      if (!r)
        j += 1
      i += 1
    }
    j
  }
}
