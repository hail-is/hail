package is.hail.services.tcp

case class ProxyTo(service: String, ns: String, port: Int, namespacedSessionId: String) {}

