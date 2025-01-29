package is.hail.utils.richUtils

import scala.collection.mutable

class RichStringBuilder(val sb: mutable.StringBuilder) extends AnyVal {
  def tsvAppend(a: Any): Unit = {
    a match {
      case null | None => sb.append("NA")
      case Some(x) => tsvAppend(x)
      case d: Double => sb.append("%.5e".format(d))
      case i: Iterable[_] =>
        var first = true
        i.foreach { x =>
          if (first)
            first = false
          else
            sb += ','
          tsvAppend(x)
        }
      case arr: Array[_] =>
        var first = true
        arr.foreach { x =>
          if (first)
            first = false
          else
            sb += ','
          tsvAppend(x)
        }
      case _ => sb.append(a)
    }
  }
}
