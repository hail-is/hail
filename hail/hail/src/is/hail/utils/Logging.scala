package is.hail.utils

import is.hail.utils.Logging.getLoggerContext

import org.apache.logging.log4j.core.LoggerContext
import org.apache.logging.log4j.scala.Logger

trait Logging {
  @transient protected lazy val logger: Logger =
    Logger(getLoggerContext.getLogger(getClass))
}

object Logging {
  // Can't say I fully understand what's going on here, but
  // spark 3.5.3 doesn't configure the active context.
  def getLoggerContext: LoggerContext =
    LoggerContext.getContext(false)
}
