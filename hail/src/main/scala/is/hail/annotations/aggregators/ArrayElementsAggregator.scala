package is.hail.annotations.aggregators

import is.hail.annotations.RegionValueBuilder
import is.hail.utils._

final case class ArrayElementsAggregator(rvAggs: Array[RegionValueAggregator]) extends RegionValueAggregator {

  override def isCommutative: Boolean = rvAggs.forall(_.isCommutative)

  // this can't be private for reflection to work
  var a: Array[Array[RegionValueAggregator]] = null

  def newInstance(): ArrayElementsAggregator = {
    ArrayElementsAggregator(rvAggs)
  }

  def broadcast(n: Int): Unit = {
    assert(a == null)
    a = new Array[Array[RegionValueAggregator]](n)
    var i = 0
    while (i < n) {
      a(i) = rvAggs.map(_.copy())
      i += 1
    }
  }

  def checkSizeOrBroadcast(n: Int): Unit = {
    if (a == null)
      broadcast(n)
    else {
      if (n != a.length)
        fatal(s"cannot apply array aggregation: size mismatch: $n, ${ a.length }")
    }
  }

  def copy(): ArrayElementsAggregator = {
    val rva = newInstance()
    if (a != null)
      rva.a = a.map(_.map(_.copy()))
    rva
  }

  def getAggs(idx: Int): Array[RegionValueAggregator] = a(idx)

  override def combOp(rva2: RegionValueAggregator): Unit = {
    val a2 = rva2.asInstanceOf[ArrayElementsAggregator].a
    if (a == null)
      a = a2
    else if (a2 != null) {
      if (a2.length != a.length)
        fatal(s"cannot apply array aggregation: size mismatch: ${ a.length }, ${ a2.length }")
      var i = 0
      while (i < a.length) {
        val aggs = a(i)
        val a2Aggs = a2(i)
        assert(aggs.length == a2Aggs.length)
        var j = 0
        while (j < aggs.length) {
          aggs(j).combOp(a2Aggs(j))
          j += 1
        }
        i += 1
      }
    }
  }

  override def result(rvb: RegionValueBuilder): Unit = {
    if (a == null)
      rvb.setMissing()
    else {
      rvb.startArray(a.length)
      a.foreach { aggs =>
        rvb.startTuple()
        aggs.foreach(_.result(rvb))
        rvb.endTuple()
      }
      rvb.endArray()
    }
  }

  override def clear(): Unit = {
    a = null
  }
}
