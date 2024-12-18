package is.hail.types

abstract class BaseType {
  final override def toString: String = {
    val sb = new StringBuilder
    pyString(sb)
    sb.result()
  }

  def toPrettyString(compact: Boolean): String = {
    val sb = new StringBuilder
    pretty(sb, 0, compact = compact)
    sb.result()
  }

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit

  def parsableString(): String = toPrettyString(compact = true)

  def pyString(sb: StringBuilder): Unit = pretty(sb, 0, compact = true)
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
