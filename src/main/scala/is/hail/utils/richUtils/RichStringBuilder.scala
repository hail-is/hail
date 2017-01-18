package is.hail.utils.richUtils

import is.hail.variant.Variant

import scala.collection.mutable

class RichStringBuilder(val sb: mutable.StringBuilder) extends AnyVal {
  def tsvAppend(a: Any) {
    a match {
      case null | None => sb.append("NA")
      case Some(x) => tsvAppend(x)
      case d: Double => sb.append(d.formatted("%.5e"))
      case v: Variant =>
        sb.append(v.contig)
        sb += ':'
        sb.append(v.start)
        sb += ':'
        sb.append(v.ref)
        sb += ':'
        sb.append(v.altAlleles.map(_.alt).mkString(","))
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
