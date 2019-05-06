package is.hail.utils

import scala.collection.mutable

class Timings(val value: mutable.Map[String, Map[String, Any]]) extends AnyVal {
  def +=(timing: (String, Map[String, Any])) { value += timing }

  def logInfo() {
    value.foreach { case (stage, timing) =>
      log.info(s"Time taken for $stage: ${ timing("readable") }")
    }
  }
}

class ExecutionTimer(context: String) {
  val timings: Timings = new Timings(mutable.Map.empty)

  def time[T](block: => T, stage: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    val nanos = t1 - t0
    val timing = Map("nano" -> nanos, "readable" -> formatTime(nanos))
    timings += s"$context -- $stage" -> timing

    result
  }
}
