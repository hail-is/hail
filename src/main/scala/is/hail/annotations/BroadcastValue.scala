package is.hail.annotations

import is.hail.expr.types.Type
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

case class BroadcastValue(value: Annotation, t: Type, sc: SparkContext) {
  lazy val broadcast: Broadcast[Annotation] = sc.broadcast(Annotation.copy(t, value))

  lazy val regionValue: RegionValue = {
    val rv = RegionValue(Region())
    val rvb = new RegionValueBuilder()
    rvb.set(rv.region)
    rvb.start(t)
    rvb.addAnnotation(t, value)
    rv.set(rv.region, rvb.end())
    rv
  }
}
