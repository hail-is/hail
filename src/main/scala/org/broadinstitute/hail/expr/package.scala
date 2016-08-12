package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, (Int, BaseType)]

  type Aggregator = TypedAggregator[Any]

  abstract class TypedAggregator[S] extends Serializable { self =>
    def zero: S
    def seqOp(x: Any, acc: S): S
    def combOp(acc1: S, acc2: S): S
    def idx: Int
    def erase: Aggregator = new TypedAggregator[Any] {
      def zero = self.zero
      def seqOp(x: Any, acc: Any) = self.seqOp(x, acc.asInstanceOf[S])
      def combOp(acc1: Any, acc2: Any) = self.combOp(acc1.asInstanceOf[S], acc2.asInstanceOf[S])
      def idx = self.idx
    }
  }

  implicit val toInt = IntNumericConversion
  implicit val toLong = LongNumericConversion
  implicit val toFloat = FloatNumericConversion
  implicit val toDouble = DoubleNumericConversion
}
