package is.hail.services

import is.hail.utils._
import org.json4s.{DefaultFormats, Formats}
import java.io.{File, FileInputStream}
import java.security.KeyStore

import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}
import org.apache.log4j.{LogManager, Logger}
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods

class NoSSLConfigFound(
  message: String,
  cause: Throwable
) extends Exception(message, cause) {
  def this() = this(null, null)

  def this(message: String) = this(message, null)
}

case class SSLConfig(
  outgoing_trust: String,
  outgoing_trust_store: String,
  incoming_trust: String,
  incoming_trust_store: String,
  key: String,
  cert: String,
  key_store: String)

package object tls {
  lazy val log: Logger = LogManager.getLogger("is.hail.tls")

  private[this] lazy val _getSSLConfig: SSLConfig = {
    var configDir = System.getenv("HAIL_SSL_CONFIG_DIR")
    if (configDir == null)
      configDir = "/ssl-config"
    val configFile = s"$configDir/ssl-config.json"
    if (!new File(configFile).isFile)
      throw new NoSSLConfigFound(s"no ssl config file found at $configFile")

    log.info(s"ssl config file found at $configFile")

    using(new FileInputStream(configFile)) { is =>
      implicit val formats: Formats = DefaultFormats
      JsonMethods.parse(is).mapField { case (k, JString(v)) => (k, JString(s"$configDir/$v")) }.extract[SSLConfig]
    }
  }

  lazy val getSSLContext: SSLContext = {
    val sslConfig = _getSSLConfig

    val pw = "dummypw".toCharArray

    val ks = KeyStore.getInstance("PKCS12")
    using(new FileInputStream(sslConfig.key_store)) { is =>
      ks.load(is, pw)
    }
    val kmf = KeyManagerFactory.getInstance("SunX509")
    kmf.init(ks, pw)

    val ts = KeyStore.getInstance("JKS")
    using(new FileInputStream(sslConfig.outgoing_trust_store)) { is =>
      ts.load(is, pw)
    }
    val tmf = TrustManagerFactory.getInstance("SunX509")
    tmf.init(ts)

    val ctx = SSLContext.getInstance("TLS")
    ctx.init(kmf.getKeyManagers, tmf.getTrustManagers, null)

    ctx
  }
}
