package is.hail.utils

import scala.collection.mutable

class ExecutionTimer(context: String) {
  val timesNanos: mutable.Map[String, Map[String, Any]] = mutable.Map.empty

  def time[T](block: => T, stage: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    val nanos = t1 - t0
    val timing = Map("nano" -> nanos, "readable" -> formatTime(nanos))
    timesNanos += s"$context -- $stage" -> timing

    result
  }

  def timings: Map[String, Map[String, Any]] = timesNanos.toMap
}
