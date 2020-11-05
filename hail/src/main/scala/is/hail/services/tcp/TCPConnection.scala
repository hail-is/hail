package is.hail.services.tcp

import java.net.Socket
import java.util.UUID

import is.hail.services.UserInfo
import org.apache.log4j.{LogManager, Logger}

class TCPConnection(
  val userInfo: UserInfo,
  val connectionId: UUID,
  val s: Socket
) {
  private[this] val log: Logger = LogManager.getLogger(s"TCPConnection")

  def log_info(msg: String): Unit = {
    log.info(s"${userInfo.email}, $connectionId, ${s.getInetAddress}: ${msg}")
  }
}
