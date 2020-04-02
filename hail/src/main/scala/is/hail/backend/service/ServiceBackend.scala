package is.hail.backend.service

object ServiceBackend {
  def apply(): ServiceBackend = {
    new ServiceBackend()
  }
}

class ServiceBackend() {
  def request(): Int = 5
}
