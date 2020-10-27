package is.hail.utils

import scala.language.implicitConversions

package object prettyPrint {
  implicit def text(t: String): Doc = Text(t)

  def space: Doc = text(" ")

  def line: Doc = Line(" ")

  def lineAlt: Doc = Line("")

  def softline: Doc = group(line)

  def softlineAlt: Doc = group(lineAlt)

  def group(body: Doc): Doc = Group(body)

  def group(body: Iterable[Doc]): Doc = group(concat(body))

  def concat(docs: Iterable[Doc]): Doc = Concat(docs)

  def concat(docs: Doc*): Doc = Concat(docs)

  def nest(i: Int, body: Doc): Doc = Indent(i, body)

  def empty: Doc = concat(Iterable.empty)

  //  def encloseSep(l: Doc, r: Doc, sep: Doc, seq: Seq[Doc]): Doc = seq match {
  //    case Seq() => concat(l, r)
  //    case Seq(s) => concat(l, s, r)
  //    case _ =>
  //      l
  //      seq.head()
  //      seq.tail.foreach { s => sep; s() }
  //      r
  //  }

  def hsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(space))

  def hsep(docs: Doc*): Doc = hsep(docs)

  def vsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(line))

  def vsep(docs: Doc*): Doc = vsep(docs)

  def sep(docs: Iterable[Doc]): Doc = group(vsep(docs))

  def sep(docs: Doc*): Doc = sep(docs)

  def fillSep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(softline))

  def list(docs: Iterable[Doc]): Doc =
    group(docs.intersperse(concat(text("("), lineAlt), concat(",", line), text(")")))

  def list(docs: Doc*): Doc = list(docs)

  def punctuate(punctuation: Doc, docs: Iterator[Doc]): Iterator[Doc] = new Iterator[Doc] {
    override def hasNext: Boolean = docs.hasNext

    override def next(): Doc = {
      val doc = docs.next()
      if (docs.hasNext)
        concat(doc, punctuation)
      else
        doc
    }
  }
}
