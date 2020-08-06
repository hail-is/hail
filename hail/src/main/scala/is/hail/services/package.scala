package is.hail

import org.apache.http.conn.HttpHostConnectException
import org.apache.log4j.{LogManager, Logger}

import scala.util.Random

package object services {
  lazy val log: Logger = LogManager.getLogger("is.hail.services")

  val RETRYABLE_HTTP_STATUS_CODES: Set[Int] = {
    val s = Set(408, 500, 502, 503, 504)
    if (System.getenv("HAIL_DONT_RETRY_500") == "1")
      s - 500
    else
      s
  }

  def sleepAndBackoff(delay: Double): Double = {
    val t = delay * Random.nextDouble()
    Thread.sleep((t * 1000).toInt)  // in ms
    math.min(delay * 2, 60.0)
  }

  def isTransientError(e: Exception): Boolean = {
    e match {
      case e: ClientResponseException =>
        RETRYABLE_HTTP_STATUS_CODES.contains(e.status)
      case e: HttpHostConnectException =>
        true
      case _ =>
        false
    }
  }

  def retryTransientErrors[T](f: => T): T = {
    var delay = 0.1
    var errors = 0
    while (true) {
      try {
        return f
      } catch {
        case e: Exception =>
          if (!isTransientError(e))
            throw e
          errors += 1
          if (errors % 10 == 0)
            log.warn(s"encountered $errors transient errors, most recent one was $e")
      }
      delay = sleepAndBackoff(delay)
    }

    throw new AssertionError("unreachable")
  }
}
