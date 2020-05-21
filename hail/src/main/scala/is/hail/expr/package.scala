package is.hail

import scala.language.implicitConversions

package object expr {
  implicit def toRichParser[T](parser: Parser.Parser[T]): RichParser[T] = new RichParser(parser)
}
