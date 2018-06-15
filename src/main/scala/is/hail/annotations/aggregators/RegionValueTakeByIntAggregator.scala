package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.Type

import scala.collection.mutable

class RegionValueTakeByIntIntAggregator(n: Int, val inputType: Type, val keyType: Type) extends RegionValueAggregator {
  val ord = keyType.ordering.toOrdering.on[(Any, Any)] { case (e, k) => k }
  var _state = new mutable.PriorityQueue[(Any, Any)]()(ord)

  def seqOp(p: (Any, Any)) {
    if (_state.length < n)
      _state += p
    else {
      if (ord.compare(p, _state.head) < 0) {
        _state.dequeue()
        _state += p
      }
    }
  }

  def seqOp(region: Region, x: Int, xm: Boolean, k: Int, km: Boolean) {
    if (!xm && !km) {
      seqOp((x, k))
    }
  }

  override def combOp(agg2: RegionValueAggregator) {
    agg2.asInstanceOf[RegionValueTakeByIntIntAggregator]._state.foreach(seqOp)
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(_state.length)
    _state.clone.dequeueAll.toArray[(Any, Any)].map(_._1).reverse.foreach { a =>
      rvb.addAnnotation(inputType, a)
    }
    rvb.endArray()
  }

  override def copy(): RegionValueTakeByIntIntAggregator = {
    val rva = new RegionValueTakeByIntIntAggregator(n, inputType, keyType)
    rva._state = _state.clone()
    rva
  }

  override def clear() {
    _state.clear()
  }
}
