package is.hail.annotations.aggregators

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.Type

import scala.collection.mutable

case class KeyedRegionValueAggregator(
  rvAgg: RegionValueAggregator,
  keyType: Type) extends RegionValueAggregator {

  var m = mutable.Map.empty[Any, RegionValueAggregator] // this can't be private for reflection to work

  def newInstance(): KeyedRegionValueAggregator = {
    KeyedRegionValueAggregator(rvAgg, keyType)
  }

  def copy(): KeyedRegionValueAggregator = {
    val rva = KeyedRegionValueAggregator(rvAgg, keyType)
    rva.m = m.map { case (k, agg) => (k, agg.copy()) }
    rva
  }

  override def combOp(agg2: RegionValueAggregator): Unit = {
    val m2 = agg2.asInstanceOf[KeyedRegionValueAggregator].m
    m2.foreach { case (k, v2) =>
      m(k) = m.get(k) match {
        case Some(v) =>
          v.combOp(v2)
          v
        case None =>
          v2
      }
    }
  }

  override def result(rvb: RegionValueBuilder): Unit = {
    rvb.startArray(m.size)
    m.foreach { case (group, rvagg) =>
      rvb.startStruct()
      rvb.addAnnotation(keyType, group)
      rvagg.result(rvb)
      rvb.endStruct()
    }
    rvb.endArray()
  }

  override def clear(): Unit = {
    m.clear()
  }
}
