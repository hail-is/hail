package is.hail.annotations

import is.hail.expr.types.Type
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

case class BroadcastValue(value: Annotation, t: Type, sc: SparkContext) {
  lazy val broadcast: Broadcast[Annotation] = sc.broadcast(value)
}
