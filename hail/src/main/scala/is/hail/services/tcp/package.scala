package is.hail.services

import java.io.{DataInputStream, DataOutputStream, OutputStream}
import java.net.{ConnectException, Socket}

import is.hail.services.tls._
import java.nio.charset.StandardCharsets
import java.util.concurrent.ExecutorService
import java.util.{Base64, UUID}

package object tcp {
  def openConnection(service: String, port: Int): (UUID, Socket) = {
    val deployConfig = DeployConfig.get
    val ns = deployConfig.getServiceNamespace(service)
    val namespacedSessionId = Tokens.get.namespaceToken(ns)
    deployConfig.location match {
      case "k8s" =>
        retryOpenConnection(
          s"${service}.${ns}",
          port,
          namespacedSessionId)
      case "gce" =>
        retryOpenConnection(
          "hail",
          5000,
          Tokens.get.namespaceToken("default"),
          Some(ProxyTo(service, ns, port, namespacedSessionId)))
      case "external" =>
        retryOpenConnection(
          "hail.is",
          5000,
          Tokens.get.namespaceToken("default"),
          Some(ProxyTo(service, ns, port, namespacedSessionId)))
    }
  }

  private[this] def retryOpenConnection(host: String,
                                        port: Int,
                                        primarySessionId: String,
                                        proxyTo: Option[ProxyTo] = None): (UUID, Socket) = {
    var s: Socket = null
    var in: DataInputStream = null
    var out: DataOutputStream = null
    var connected = false
    while (!connected) {
      s = socket(host, port)
      in = new DataInputStream(s.getInputStream)
      out = new DataOutputStream(s.getOutputStream)

      proxyTo match {
        case None =>
          writeSessionIds(out, primarySessionId, None)

        case Some(ProxyTo(service, ns, port, namespacedSessionId)) =>
          writeSessionIds(out, primarySessionId, Some(namespacedSessionId))

          out.writeInt(ns.length)
          out.write(ns.getBytes(StandardCharsets.UTF_8))

          out.writeInt(service.length)
          out.write(service.getBytes(StandardCharsets.UTF_8))

          out.writeShort(port)
          out.flush()
      }

      val isSuccess = in.read()
      if (isSuccess == -1) {
        log.info("end of file encountered before reading anything, retrying connection")
      } else if (isSuccess != 1) {
        throw new HailTCPConnectionError(s"${host}:${port} ${isSuccess}")
      } else {
        connected = true
      }
    }

    val connectionIdMostSignificant = in.readLong()
    val connectionIdLeastSignificant = in.readLong()

    (new UUID(connectionIdMostSignificant, connectionIdLeastSignificant), s)
  }

  private[this] val sessionIdDecoder = Base64.getUrlDecoder
  def sessionIdDecodeFromStr(id: String): Array[Byte] = sessionIdDecoder.decode(id)

  private[this] val sessionIdEncoder = Base64.getUrlEncoder
  def sessionIdEncodeToStr(id: Array[Byte]): String = sessionIdEncoder.encodeToString(id)

  private[this] def writeSessionIds(out: OutputStream,
                                    defaultSessionId: String,
                                    namespacedSessionId: Option[String]): Unit = {
    val defaultSessionIdBytes = sessionIdDecodeFromStr(defaultSessionId)
    assert(defaultSessionIdBytes.length == 32)
    out.write(defaultSessionIdBytes)
    namespacedSessionId match {
      case None =>
        out.write(new Array[Byte](32))
      case Some(namespacedSessionId) =>
        val namespacedSessionIdBytes = sessionIdDecodeFromStr(namespacedSessionId)
        assert(namespacedSessionIdBytes.length == 32)
        out.write(namespacedSessionIdBytes)
    }
  }

  private[this] def socket(host: String, port: Int): Socket = {
    var s: Socket = null
    var attempts = 0
    while (s == null) {
      try {
        s = getSSLContext.getSocketFactory().createSocket(host, port)
      } catch {
        case e: ConnectException =>
          if (attempts % 10 == 0) {
            log.warn(s"retrying socket connect to ${host}:${port} after receiving ${e}")
          }
          attempts += 1
      }
    }
    assert(s != null)
    s
  }
}
