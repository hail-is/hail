package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types._

trait RegionValueAggregator extends Serializable {

  def combOp(agg2: RegionValueAggregator): Unit

  def result(rvb: RegionValueBuilder)

  def newInstance(): RegionValueAggregator

  def copy(): RegionValueAggregator

  def clear(): Unit
}
