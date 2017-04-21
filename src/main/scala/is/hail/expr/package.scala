package is.hail

import scala.collection.mutable
import scala.language.implicitConversions
import is.hail.asm4s.{CM, Code, TypeInfo}

package object expr extends HailRepFunctions {
  type SymbolTable = Map[String, (Int, Type)]
  def emptySymTab = Map.empty[String, (Int, Type)]

  type Aggregator = TypedAggregator[Any]

  abstract class TypedAggregator[+S] extends Serializable {
    def seqOp(x: Any): Unit

    def combOp(agg2: this.type): Unit

    def result: S

    def copy(): TypedAggregator[S]
  }

  implicit def toRichParser[T](parser: Parser.Parser[T]): RichParser[T] = new RichParser(parser)

  type CPS[T] = (T => Unit) => Unit
  type CMCodeCPS[T] = (Code[T] => CM[Code[Unit]]) => CM[Code[Unit]]

  def ec(): CM[EvalContext] =
    CM.st().map(_.asInstanceOf[EvalContext])
  def currentSymbolTable(): CM[SymbolTable] = for (
    ec <- CM.st().map(_.asInstanceOf[EvalContext])
  ) yield ec.st

  def ecNewPosition(): CM[(Int, mutable.ArrayBuffer[Any])] = for (
    ec <- CM.st().map(_.asInstanceOf[EvalContext])
  ) yield {
    val idx = ec.a.length
    val localA = ec.a
    localA += null
    (idx, localA)
  }

  // returns a thunk that looks up the result of this aggregation
  def addAggregation(lhs: AST, agg: Aggregator): CM[() => Any] =  for (
    ec <- CM.st().map(_.asInstanceOf[EvalContext])
  ) yield {
    val b = RefBox(null)
    ec.aggregations += ((b, lhs.runAggregator(ec), agg))
    () => b.v
  }

  def castCode[T](ti: TypeInfo[T], c: CM[Code[AnyRef]]): (TypeInfo[T], CM[Code[T]]) =
    (ti, c.map(Code.checkcast(_)(ti)))

  def unsafeCastCode[T](ti: TypeInfo[T], c: CM[Code[AnyRef]]) = (ti, c.asInstanceOf[CM[Code[T]]])

}
