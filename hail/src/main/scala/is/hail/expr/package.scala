package is.hail

import scala.language.implicitConversions

package expr {
  abstract class TypedAggregator[+S] extends Serializable {
    def seqOp(x: Any): Unit

    def combOp(agg2: this.type): Unit

    def result: S

    def copy(): TypedAggregator[S]
  }
}

package object expr {
  implicit def toRichParser[T](parser: Parser.Parser[T]): RichParser[T] = new RichParser(parser)
}