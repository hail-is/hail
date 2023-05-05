package is.hail.backend.service

import is.hail.HailContext
import org.apache.log4j.{LogManager, PropertyConfigurator}

import java.util.Properties

object Main {
  val WORKER = "worker"
  val DRIVER = "driver"

  def main(argv: Array[String]): Unit = {
    argv(3) match {
      case WORKER => Worker.main(argv)
      case DRIVER => ServiceBackendSocketAPI2.main(argv)
      case kind => throw new RuntimeException(s"unknown kind: ${kind}")
    }
  }
}
