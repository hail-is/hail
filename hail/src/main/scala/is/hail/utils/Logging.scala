package is.hail.utils

import org.apache.log4j.{Logger, LogManager}

object LogHelper {
  // exposed more directly for generated code
  def logInfo(msg: String): Unit = log.info(msg)
  def warning(msg: String): Unit = consoleLog.warn(msg)
  def consoleInfo(msg: String): Unit = info(msg)
}

trait Logging {
  @transient private var logger: Logger = _
  @transient private var consoleLogger: Logger = _

  def log: Logger = {
    if (logger == null)
      logger = LogManager.getRootLogger
    logger
  }

  def consoleLog: Logger = {
    if (consoleLogger == null)
      consoleLogger = LogManager.getLogger("Hail")
    consoleLogger
  }

  def info(msg: String) {
    consoleLog.info(msg)
  }

  def info(msg: String, t: Truncatable) {
    val (screen, logged) = t.strings
    if (screen == logged)
      consoleLog.info(format(msg, screen))
    else {
      // writes twice to the log file, but this isn't a big problem
      consoleLog.info(format(msg, screen))
      log.info(format(msg, logged))
    }
  }

  def warn(msg: String) {
    consoleLog.warn(msg)
  }

  def warn(msg: String, t: Truncatable) {
    val (screen, logged) = t.strings
    if (screen == logged)
      consoleLog.warn(format(msg, screen))
    else {
      // writes twice to the log file, but this isn't a big problem
      consoleLog.warn(format(msg, screen))
      log.warn(format(msg, logged))
    }
  }

  def error(msg: String) {
    consoleLog.error(msg)
  }
}
