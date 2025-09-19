package is.hail.utils

import java.util.Properties

import org.apache.log4j.{LogManager, Logger, PropertyConfigurator}

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

  def info(msg: String): Unit =
    consoleLog.info(msg)

  def info(msg: String, t: Truncatable): Unit = {
    val (screen, logged) = t.strings
    if (screen == logged)
      consoleLog.info(format(msg, screen))
    else {
      // writes twice to the log file, but this isn't a big problem
      consoleLog.info(format(msg, screen))
      log.info(format(msg, logged))
    }
  }

  def warn(msg: String): Unit =
    consoleLog.warn(msg)

  def warn(msg: String, t: Truncatable): Unit = {
    val (screen, logged) = t.strings
    if (screen == logged)
      consoleLog.warn(format(msg, screen))
    else {
      // writes twice to the log file, but this isn't a big problem
      consoleLog.warn(format(msg, screen))
      log.warn(format(msg, logged))
    }
  }

  def error(msg: String): Unit =
    consoleLog.error(msg)
}

object Logging {

  val LogFormat: String =
    "%d{yyyy-MM-dd HH:mm:ss.SSS} %c{1}: %p: %m%n"

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean): Unit = {
    org.apache.log4j.helpers.LogLog.setInternalDebugging(true)
    org.apache.log4j.helpers.LogLog.setQuietMode(false)
    val logProps = new Properties()

    // uncomment to see log4j LogLog output:
    // logProps.put("log4j.debug", "true")
    logProps.put("log4j.rootLogger", "INFO, logfile")
    logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
    logProps.put("log4j.appender.logfile.append", append.toString)
    logProps.put("log4j.appender.logfile.file", logFile)
    logProps.put("log4j.appender.logfile.threshold", "INFO")
    logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
    logProps.put("log4j.appender.logfile.layout.ConversionPattern", LogFormat)

    if (!quiet) {
      logProps.put("log4j.logger.Hail", "INFO, HailConsoleAppender, HailSocketAppender")
      logProps.put("log4j.appender.HailConsoleAppender", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.HailConsoleAppender.target", "System.err")
      logProps.put("log4j.appender.HailConsoleAppender.layout", "org.apache.log4j.PatternLayout")
    } else
      logProps.put("log4j.logger.Hail", "INFO, HailSocketAppender")

    logProps.put("log4j.appender.HailSocketAppender", "is.hail.utils.StringSocketAppender")
    logProps.put("log4j.appender.HailSocketAppender.layout", "org.apache.log4j.PatternLayout")

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)
  }

}
