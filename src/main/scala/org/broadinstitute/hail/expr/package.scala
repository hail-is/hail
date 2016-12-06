package org.broadinstitute.hail

package object expr {
  type SymbolTable = Map[String, (Int, Type)]
  def emptySymTab = Map.empty[String, (Int, Type)]

  type Aggregator = TypedAggregator[Any]

  abstract class TypedAggregator[+S] extends Serializable {
    def seqOp(x: Any): Unit

    def combOp(agg2: this.type): Unit

    def result: S

    def copy(): TypedAggregator[S]
  }
}
