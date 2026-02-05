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
  def getMissingIndexAndCount(req: IndexedSeq[Boolean]): (IndexedSeq[Int], Int) = {
    val scan = req.scanLeft(0) { case (j, r) =>
      j + (if (r) 0 else 1)
    }
    (scan.init, scan.last)
  }
}
