package is.hail.annotations.aggregators

import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.ir._
import is.hail.annotations._
import is.hail.expr.RegionValueAggregator

class FractionAggregator extends RegionValueAggregator {
  private var _num = 0L
  private var _denom = 0L


  def result =
    if (_denom == 0L)
      null
    else
      _num.toDouble / _denom

  def seqOp(x: Any) {
    val r = f(x)
    _denom += 1
    if (r.asInstanceOf[Boolean])
      _num += 1
  }

  def combOp(agg2: this.type) {
    _num += agg2._num
    _denom += agg2._denom
  }

  def copy() = new FractionAggregator(f)
}
