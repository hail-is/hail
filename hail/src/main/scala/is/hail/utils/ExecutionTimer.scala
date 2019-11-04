package is.hail.utils

import scala.collection.mutable

class Timings(val value: mutable.Map[String, Long], ord: mutable.ArrayBuffer[String]) {
  def +=(timing: (String, Long)) {
    ord += timing._1
    value += timing
  }

  def logInfo() {
    ord.foreach { stage =>
      val timing = value(stage)
      log.info(s"Time taken for $stage: ${ formatTime(timing) }")
    }
  }
}

class ExecutionTimer {
  val timings: Timings = new Timings(mutable.Map.empty, mutable.ArrayBuffer.empty)

  def time[T](block: => T, operation: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    val nanos = t1 - t0
    timings += s"$operation" -> nanos

    result
  }
}
