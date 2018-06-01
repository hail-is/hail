package is.hail.annotations.aggregators

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.Type

import scala.collection.mutable

case class KeyedRegionValueAggregator[Agg <: RegionValueAggregator](
  rvAgg: Agg,
  keyTyp: Type,
  aggTyp: Type,
  resultTyp: Type) extends RegionValueAggregator {

  val m = mutable.Map.empty[Any, Agg] // this can't be private for reflection to work

  override def copy(): RegionValueAggregator = {
    KeyedRegionValueAggregator(rvAgg, keyTyp, aggTyp, resultTyp)
  }

  override def combOp(agg2: RegionValueAggregator): Unit = {
    val m2 = agg2.asInstanceOf[KeyedRegionValueAggregator[Agg]].m
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
      rvb.addAnnotation(keyTyp, group)
      rvagg.result(rvb)
      rvb.endStruct()
    }
    rvb.endArray()
  }

  override def clear(): Unit = {
    m.clear()
  }
}
