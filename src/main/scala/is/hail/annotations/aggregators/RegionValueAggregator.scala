package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types._

trait RegionValueAggregator extends Serializable {
  var inputTyp: Type = _
  var resultTyp: Type = _

  def setTypes(input: Type, result: Type) {
    inputTyp = input
    resultTyp = result
  }

  def combOp(agg2: RegionValueAggregator): Unit

  def result(rvb: RegionValueBuilder)

  def copy(): RegionValueAggregator

  def clear(): Unit
}
