package is.hail.annotations.aggregators

import java.util

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.Type

import scala.collection.JavaConverters._
import scala.language.implicitConversions

case class KeyedRegionValueAggregator(
  rvAgg: RegionValueAggregator,
  keyType: Type) extends RegionValueAggregator {

  var m = new java.util.HashMap[Any, RegionValueAggregator]() // this can't be private for reflection to work

  def newInstance(): KeyedRegionValueAggregator = {
    KeyedRegionValueAggregator(rvAgg, keyType)
  }

  def copy(): KeyedRegionValueAggregator = {
    val rva = KeyedRegionValueAggregator(rvAgg, keyType)
    rva.m = new util.HashMap[Any, RegionValueAggregator](m.asScala.mapValues(_.copy()).asJava)
    rva
  }

  override def combOp(rva2: RegionValueAggregator): Unit = {
    val m2 = rva2.asInstanceOf[KeyedRegionValueAggregator].m
    m2.asScala.foreach { case (k, agg2) =>
      val agg = m.get(k)
      if (agg == null)
        m.put(k, agg2)
      else
        agg.combOp(agg2)
    }
  }

  override def result(rvb: RegionValueBuilder): Unit = {
    val sorted = m.asScala.toArray.sortBy(f => f._1)(keyType.ordering.toOrdering)
    rvb.startArray(m.size)
    sorted.foreach { case (group, rvagg) =>
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
