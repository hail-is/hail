package is.hail.utils

import org.apache.logging.log4j.{LogManager, Logger}

object LogHelper {
  // exposed more directly for generated code
  def logInfo(msg: String): Unit = log.info(msg)
  def warning(msg: String): Unit = log.warn(msg)
  def consoleInfo(msg: String): Unit = info(msg)
}

trait Logging {
  @transient lazy val log: Logger =
    LogManager.getLogger(getClass.getName.stripSuffix("$"))

  def info(msg: String): Unit =
    log.info(msg)

  def warn(msg: String): Unit =
    log.warn(msg)

  def error(msg: String): Unit =
    log.error(msg)
}
