package is.hail

package object expr {
  implicit def toRichParser[T](parser: Parser.Parser[T]): RichParser[T] = new RichParser(parser)
}
