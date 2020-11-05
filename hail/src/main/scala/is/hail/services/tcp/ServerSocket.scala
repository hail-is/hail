package is.hail.services.tcp

import java.io.{Closeable, DataInputStream, DataOutputStream}
import java.net.Socket
import java.util.UUID
import java.util.concurrent.ExecutorService

import is.hail.services.{ClientResponseException, DeployConfig, Requester, UserInfo, tcp}
import is.hail.services.tls.getSSLContext
import javax.net.ssl.SSLServerSocket
import org.apache.http.client.methods.HttpGet
import org.apache.log4j.{LogManager, Logger}

class ServerSocket(port: Int, executor: ExecutorService) extends Closeable {
  lazy val log: Logger = LogManager.getLogger("ServerSocket")

  private[this] val ss = {
    val ssl = getSSLContext
    val ssf = ssl.getServerSocketFactory
    ssf.createServerSocket(port).asInstanceOf[SSLServerSocket]
  }

  private[this] val auth = new Requester("auth")
  private[this] val authBaseUrl = DeployConfig.get.baseUrl("auth")

  class GetUserInfo(
    private[this] val s: Socket,
    private[this] val next: (TCPConnection) => Unit
  ) extends Runnable {
    def run(): Unit = {
      log.info(s"accepted connection from ${s.getInetAddress}")
      var uuid: UUID = null
      try {
        val in = new DataInputStream(s.getInputStream)
        val out = new DataOutputStream(s.getOutputStream)

        val sessionIdBytes = new Array[Byte](32)
        in.read(sessionIdBytes)
        val sessionId = tcp.sessionIdEncodeToStr(sessionIdBytes)
        in.skipBytes(32) // internal auth is only for routers and gateways

        log.info(s"checking session id ${sessionId} from ${s.getInetAddress}")

        val userInfo = try {
          val req = new HttpGet(s"$authBaseUrl/api/v1alpha/userinfo")
          req.addHeader("Authorization", s"Bearer ${sessionId}")
          UserInfo.fromJValue(auth.request(req, addAuthHeaders = false))
        } catch {
          case exc: ClientResponseException if exc.status == 401 =>
            log.info("invalid credentials", exc)
            out.write(0)
            s.close()
            return
          case exc: Exception =>
            log.error("unexpected exception getting user info", exc)
            out.write(0)
            s.close()
            return
        }

        uuid = UUID.randomUUID()
        out.write(1)
        out.writeLong(uuid.getMostSignificantBits)
        out.writeLong(uuid.getLeastSignificantBits)
        out.flush()

        val conn = new TCPConnection(userInfo, uuid, s)
        conn.log_info(s"authenticated connection, calling handler")
        next(conn)
      } catch {
        case exc: Exception =>
          log.error(s"exception during handling ${uuid}")
          s.close()
      }
    }
  }

  def serveForever(handle: (TCPConnection) => Unit): Unit = {
    while (true) {
      val s = ss.accept()
      executor.submit(new GetUserInfo(s, handle))
    }
  }

  def close(): Unit = {
    ss.close()
  }

}
