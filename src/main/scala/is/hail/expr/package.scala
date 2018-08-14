package is.hail

import is.hail.expr.types.Type
import scala.language.implicitConversions

package object expr extends HailRepFunctions {
  type SymbolTable = Map[String, (Int, Type)]

  def emptySymTab = Map.empty[String, (Int, Type)]

  def hailType[T: HailRep]: Type = implicitly[HailRep[T]].typ

  type Aggregator = TypedAggregator[Any]

  abstract class TypedAggregator[+S] extends Serializable {
    def seqOp(x: Any): Unit

    def combOp(agg2: this.type): Unit

    def result: S

    def copy(): TypedAggregator[S]
  }

  implicit def toRichParser[T](parser: Parser.Parser[T]): RichParser[T] = new RichParser(parser)
}
