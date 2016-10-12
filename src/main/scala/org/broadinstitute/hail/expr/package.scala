package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, (Int, BaseType)]
  type Aggregator = TypedAggregator[_, _]

  abstract class TypedAggregator[+S, E] extends Serializable {

    val f: (Any) => Any

    def merge(x: E): Unit

    def seqOp(x: Any): Unit = {
      val r = f(x)
      if (r != null)
        merge(r.asInstanceOf[E])
    }

    def combOp(agg2: TypedAggregator[_, _]): Unit

    def result: S

    def copy(): TypedAggregator[S, E]

    def idx: Int
  }

  implicit val toInt = IntNumericConversion
  implicit val toLong = LongNumericConversion
  implicit val toFloat = FloatNumericConversion
  implicit val toDouble = DoubleNumericConversion
}
