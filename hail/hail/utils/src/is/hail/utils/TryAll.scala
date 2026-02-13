package is.hail.utils

import scala.util.{Failure, Success, Try}

object TryAll {
  def apply[K](f: => K): Try[K] =
    try
      Success(f)
    catch {
      case e: Throwable => Failure(e)
    }
}
