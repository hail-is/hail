package is.hail.utils

package object prettyPrint {
  implicit def text(t: String): Doc = Text(t)

  def space: Doc = text(" ")

  // Prints as a newline, followed by the current indentation, unless part of a group
  // printing in flat mode, in which case it prints as a space.
  def line: Doc = Line(" ")

  // Like line, but prints as either a newline or nothing.
  def lineAlt: Doc = Line("")

  // Prints as a newline, unless that would cause an overfull line,
  // in which case it prints as a space.
  def softline: Doc = group(line)

  // Like softline, but prints as either a newline or nothing.
  def softlineAlt: Doc = group(lineAlt)

  // Prints body in one of two ways:
  // 1. Flattened, converting all contained `line`s (resp. `lineAlt`s)
  //    to spaces (resp. empty strings), including in nested groups.
  // 2. Or, if 1. would result in an overfull line, prints contained
  //    `line`s and `lineAlt`s as newlines. Nested groups may still
  //    be printed either flattened or not, as space allows.
  def group(body: Doc*): Doc =
    if (body.size == 1) Group(body.head) else group(body)

  def group(body: Iterable[Doc]): Doc = group(concat(body))

  // Simple concatenation of documents.
  def concat(docs: Iterable[Doc]): Doc = Concat(docs)

  def concat(docs: Doc*): Doc = Concat(docs)

  // All newlines in body have indentation increased by i.
  def nest(i: Int, body: Doc): Doc = Indent(i, body)

  def empty: Doc = concat(Iterable.empty)

  // Concatenate `docs` separated by (unconditional) spaces
  def hsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(space))

  def hsep(docs: Doc*): Doc = hsep(docs)

  // Concatenate `docs` separated by newlines (unless flattened by an enclocing group).
  def vsep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(line))

  def vsep(docs: Doc*): Doc = vsep(docs)

  // Concatenate `docs` separated by spaces, if that would fit, or else by newlines.
  def sep(docs: Iterable[Doc]): Doc = group(vsep(docs))

  def sep(docs: Doc*): Doc = sep(docs)

  // Concatenate `docs` separated by spaces, wrapping to a new line when the next
  // doc doesn't fit.
  def fillSep(docs: Iterable[Doc]): Doc = concat(docs.intersperse(softline))

  def fillSep(docs: Doc*): Doc = fillSep(docs)

  // Print `docs` as a paren-delimited list, either all on one line separated by spaces,
  // or with each element on a separate line. Add indentation with a containing `nest`, e.g.
  // nest(2, list(docs)).
  def list(docs: Iterable[Doc]): Doc =
    group(concat(docs.intersperse[Doc]("(", line, ")")))

  def list(docs: Doc*): Doc = list(docs)

  // Print `docs` as a paren-delimited list, with elements separated by spaces, wrapping to a new
  // line as needed. Add indentation with a containing `nest`, e.g. nest(2, fillList(docs)).
  def fillList(docs: Iterable[Doc]): Doc =
    concat(text("("), softlineAlt, fillSep(docs), text(")"))

  def fillList(docs: Doc*): Doc = fillList(docs)
}
