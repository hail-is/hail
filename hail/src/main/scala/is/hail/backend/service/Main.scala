package is.hail.backend.service

import is.hail.HailContext
import org.apache.log4j.{LogManager, PropertyConfigurator}

import java.util.Properties

object Main {
  val WORKER = "worker"
  val DRIVER = "driver"

  def configureLogging(logFile: String): Unit = {
    val logProps = new Properties()

    logProps.put("log4j.rootLogger", "INFO, logfile")
    logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
    logProps.put("log4j.appender.logfile.append", true.toString)
    logProps.put("log4j.appender.logfile.file", logFile)
    logProps.put("log4j.appender.logfile.threshold", "INFO")
    logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
    logProps.put("log4j.appender.logfile.layout.ConversionPattern", HailContext.logFormat)

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)
  }

  def main(argv: Array[String]): Unit = {
    val logFile = argv(1)
    configureLogging(logFile)

    argv(3) match {
      case WORKER => Worker.main(argv)
      case DRIVER => ServiceBackendSocketAPI2.main(argv)
      case kind => throw new RuntimeException(s"unknown kind: ${kind}")
    }
  }
}
