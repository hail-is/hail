package is.hail.backend.service


object Main {
  val WORKER = "worker"
  val DRIVER = "driver"

  def main(argv: Array[String]): Unit =
    argv(3) match {
      case WORKER => Worker.main(argv)
      case DRIVER => ServiceBackendAPI.main(argv)
      case kind => throw new RuntimeException(s"unknown kind: $kind")
    }
}
