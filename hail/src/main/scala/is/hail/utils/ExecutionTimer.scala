package is.hail.utils

class ExecutionTimer() {
  var timingsNanoSecs: Map[String, Long] = Map.empty

  def time[T](block: => T, name: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    timingsNanoSecs += name -> (t1 - t0)

    result
  }

  def times: Map[String, Long] = timingsNanoSecs
}
