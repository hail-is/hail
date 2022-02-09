package is.hail.backend.service

object Main {
  val WORKER = 1
  val DRIVER = 2

  def main(argv: Array[String]): Unit = {
    argv(0) match {
      case WORKER => Worker.main(argv)
      case DRIVER => ServiceBackendSocketAPI2.main(argv)
      case kind => throw new RuntimeException(s"unknown kind: ${kind}")
    }
  }
}
