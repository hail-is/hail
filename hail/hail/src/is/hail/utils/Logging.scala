package is.hail.utils

import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.scala.Logger

trait Logging {
  protected lazy val logger: Logger =
    Logger(LogManager.getContext.getLogger(getClass))
}
