package is.hail.expr

import is.hail.expr.types._

import scala.language.existentials
import scala.util.parsing.input.Positional

case class EvalContext(st: SymbolTable) {
  st.foreach {
    case (name, (i, t: TAggregable)) =>
      require(t.symTab.exists { case (_, (j, t2)) => j == i && t2 == t.elementType },
        s"did not find binding for type ${ t.elementType } at index $i in agg symbol table for `$name'")
    case _ =>
  }
}

object EvalContext {
  def apply(args: (String, Type)*): EvalContext = {
    EvalContext(args.zipWithIndex
      .map { case ((name, t), i) => (name, (i, t)) }
      .toMap)
  }
}

object DoubleNumericConversion {
  def to(numeric: Any): Double = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
    case d: Double => d
  }
}

case class Positioned[T](x: T) extends Positional
