package is.hail.services

import java.io._
import java.net.Socket
import java.security.KeyStore
import java.util.Base64

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.types.physical._
import is.hail.io._
import is.hail.utils._
import org.apache.log4j.Logger
import javax.net.ssl._;
import scala.language.implicitConversions

package object shuffler {
  val shuffleBufferSpec = BufferSpec.unblockedUncompressed

  /**
    * The following creates a server key and cert, client key and cert, a server
    * key and trust store, and a client key and trust store. The server trusts
    * only the client and the client trusts only the server.
    *
    * openssl req -x509 -newkey rsa:4096 -keyout server-key.pem -out server-cert.pem -days 365 -subj '/CN=localhost'
    * openssl req -x509 -newkey rsa:4096 -keyout client-key.pem -out client-cert.pem -days 365 -subj '/CN=localhost'
    *
    * openssl pkcs12 -export -out server-keystore.p12 -inkey server-key.pem -in server-cert.pem
    * keytool -import -alias client-cert -file client-cert.pem -keystore server-truststore.p12
    *
    * openssl pkcs12 -export -out client-keystore.p12 -inkey client-key.pem -in client-cert.pem
    * keytool -import -alias client-cert -file server-cert.pem -keystore client-truststore.p12
    *
    **/
  def sslContext(
    keyStorePath: String,
    keyStorePassPhrase: String,
    keyStoreType: String,
    trustStorePath: String,
    trustStorePassPhrase: String,
    trustStoreType: String
  ): SSLContext = sslContext(
    new FileInputStream(keyStorePath), keyStorePassPhrase, keyStoreType,
    new FileInputStream(trustStorePath), trustStorePassPhrase, trustStoreType)

  def sslContext(
    keyStoreInputStream: InputStream,
    keyStorePassPhrase: String,
    keyStoreType: String,
    trustStoreInputStream: InputStream,
    trustStorePassPhrase: String,
    trustStoreType: String
  ): SSLContext = {
    val ctx = SSLContext.getInstance("TLS")
    val kmf = KeyManagerFactory.getInstance("SunX509")
    val ks = KeyStore.getInstance(keyStoreType)
    ks.load(keyStoreInputStream, keyStorePassPhrase.toCharArray())
    kmf.init(ks, keyStorePassPhrase.toCharArray())
    val tmf = TrustManagerFactory.getInstance("SunX509")
    val ts = KeyStore.getInstance(trustStoreType)
    ts.load(trustStoreInputStream, trustStorePassPhrase.toCharArray())
    tmf.init(ts)
    ctx.init(kmf.getKeyManagers(), tmf.getTrustManagers(), null)
    ctx
  }

  def rvstr(pt: PType, off: Long): String =
    UnsafeRow.read(pt, null, off).toString

  def writeRegionValueArray(
    encoder: Encoder,
    values: Array[Long]
  ): Unit = {
    var i = 0
    while (i < values.length) {
      encoder.writeByte(1)
      encoder.writeRegionValue(values(i))
      i += 1
    }
    encoder.writeByte(0)
  }

  def readRegionValueArray(
    region: Region,
    decoder: Decoder,
    sizeHint: Int = ArrayBuilder.defaultInitialCapacity
  ): Array[Long] = {
    val ab = new ArrayBuilder[Long](sizeHint)

    var hasNext = decoder.readByte()
    while (hasNext == 1) {
      ab += decoder.readRegionValue(region)
      hasNext = decoder.readByte()
    }
    assert(hasNext == 0, hasNext)

    ab.result()
  }

  private[this] val b64encoder = Base64.getEncoder()

  def uuidToString(uuid: Array[Byte]): String =
    b64encoder.encodeToString(uuid)

  def uuidToString(uuid: Code[Array[Byte]]): Code[String] =
    Code.invokeScalaObject1[Array[Byte], String](getClass, "uuidToString", uuid)
}
