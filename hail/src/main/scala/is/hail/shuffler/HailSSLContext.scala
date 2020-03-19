package is.hail.shuffler

import java.io._
import java.nio.charset.StandardCharsets
import java.security.KeyStore;
import javax.net._
import javax.net.ssl._
import javax.security.cert.X509Certificate;

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

object HailSSLContext {
  def apply(
    keyStorePath: String,
    keyStorePassPhrase: String,
    trustStorePath: String,
    trustStorePassPhrase: String
  ): SSLContext = apply(
    new FileInputStream(keyStorePath), keyStorePassPhrase,
    new FileInputStream(trustStorePath), trustStorePassPhrase)

  def apply(
    keyStoreInputStream: InputStream,
    keyStorePassPhrase: String,
    trustStoreInputStream: InputStream,
    trustStorePassPhrase: String
  ): SSLContext = {
    val ctx = SSLContext.getInstance("TLS")
    val kmf = KeyManagerFactory.getInstance("SunX509")
    val ks = KeyStore.getInstance("PKCS12")
    ks.load(keyStoreInputStream, keyStorePassPhrase.toCharArray())
    kmf.init(ks, keyStorePassPhrase.toCharArray())
    val tmf = TrustManagerFactory.getInstance("SunX509")
    val ts = KeyStore.getInstance("JKS")
    ts.load(trustStoreInputStream, trustStorePassPhrase.toCharArray())
    tmf.init(ts)
    ctx.init(kmf.getKeyManagers(), tmf.getTrustManagers(), null)
    ctx
  }
}
