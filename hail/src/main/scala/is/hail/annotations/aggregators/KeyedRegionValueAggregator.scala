package is.hail.annotations.aggregators

import java.util

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.virtual.Type

import scala.collection.JavaConverters._

case class KeyedRegionValueAggregator(
  rvAggs: Array[RegionValueAggregator],
  keyType: Type) extends RegionValueAggregator {

  override def isCommutative: Boolean = rvAggs.forall(_.isCommutative)

  var m = new java.util.HashMap[Any, Array[RegionValueAggregator]]() // this can't be private for reflection to work

  def newInstance(): KeyedRegionValueAggregator = {
    KeyedRegionValueAggregator(rvAggs, keyType)
  }

  def copy(): KeyedRegionValueAggregator = {
    val rva = KeyedRegionValueAggregator(rvAggs, keyType)
    rva.m = new util.HashMap[Any, Array[RegionValueAggregator]](m.asScala.mapValues(_.map(_.copy())).asJava)
    rva
  }

  def getAggs(key: Any): Array[RegionValueAggregator] = {
    if (!m.containsKey(key))
      m.put(key, rvAggs.map(_.copy()))
    m.get(key)
  }

  override def combOp(rva2: RegionValueAggregator): Unit = {
    val m2 = rva2.asInstanceOf[KeyedRegionValueAggregator].m
    m2.asScala.foreach { case (k, agg2) =>
      val agg = m.get(k)
        if (agg == null)
          m.put(k, agg2)
        else {
          var i = 0
          while(i < agg.length) {
            agg(i).combOp(agg2(i))
            i += 1
          }
        }
    }
  }

  override def result(rvb: RegionValueBuilder): Unit = {
    val sorted = m.asScala.toArray.sortBy(f => f._1)(keyType.ordering.toOrdering)
    rvb.startArray(m.size)
    sorted.foreach { case (group, rvagg) =>
      rvb.startStruct()
      rvb.addAnnotation(keyType, group)
      rvb.startTuple()
      var i = 0
      while(i < rvagg.length) {
        rvagg(i).result(rvb)
        i += 1
      }
      rvb.endTuple()
      rvb.endStruct()
    }
    rvb.endArray()
  }

  override def clear(): Unit = {
    m.clear()
  }
}
