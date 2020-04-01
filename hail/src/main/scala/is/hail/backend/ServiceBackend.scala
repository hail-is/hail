package is.hail.backend

object ServiceBackend {
  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class ServiceBackend() {
  def request(): Int = 5
}
