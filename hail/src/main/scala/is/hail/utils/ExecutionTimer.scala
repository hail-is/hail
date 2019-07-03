package is.hail.utils

import scala.collection.mutable

class Timings(val value: mutable.Map[String, Map[String, Any]], ord: mutable.ArrayBuffer[String]) {
  def +=(timing: (String, Map[String, Any])) {
    ord += timing._1
    value += timing
  }

  def logInfo() {
    ord.foreach { stage =>
      val timing = value(stage)
      log.info(s"Time taken for $stage: ${ timing("readable") }")
    }
  }
}

class ExecutionTimer(val context: String) {
  val timings: Timings = new Timings(mutable.Map.empty, mutable.ArrayBuffer.empty)

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
