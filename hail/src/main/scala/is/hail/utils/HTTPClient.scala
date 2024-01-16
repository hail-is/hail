package is.hail.utils
import java.io.{InputStream, OutputStream}
import java.net.{HttpURLConnection, URL}
import java.nio.charset.StandardCharsets

import org.apache.commons.io.output.ByteArrayOutputStream

object HTTPClient {
  def post[T](
    url: String,
    contentLength: Int,
    writeBody: OutputStream => Unit,
    readResponse: InputStream => T = (_: InputStream) => (),
    chunkSize: Int = 0,
  ): T = {
    val conn = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    conn.setRequestMethod("POST")
    if (chunkSize > 0)
      conn.setChunkedStreamingMode(chunkSize)
    conn.setDoOutput(true);
    conn.setRequestProperty("Content-Length", Integer.toString(contentLength))
    using(conn.getOutputStream())(writeBody)
    assert(
      200 <= conn.getResponseCode() && conn.getResponseCode() < 300,
      s"POST $url ${conn.getResponseCode()} ${using(conn.getErrorStream())(fullyReadInputStreamAsString)}",
    )
    val result = using(conn.getInputStream())(readResponse)
    conn.disconnect()
    result
  }

  def get[T](
    url: String,
    readResponse: InputStream => T,
  ): T = {
    val conn = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    conn.setRequestMethod("GET")
    assert(
      200 <= conn.getResponseCode() && conn.getResponseCode() < 300,
      s"GET $url ${conn.getResponseCode()} ${using(conn.getErrorStream())(fullyReadInputStreamAsString)}",
    )
    val result = using(conn.getInputStream())(readResponse)
    conn.disconnect()
    result
  }

  def delete(
    url: String,
    readResponse: InputStream => Unit = (_: InputStream) => (),
  ): Unit = {
    val conn = new URL(url).openConnection().asInstanceOf[HttpURLConnection]
    conn.setRequestMethod("DELETE")
    assert(
      200 <= conn.getResponseCode() && conn.getResponseCode() < 300,
      s"DELETE $url ${conn.getResponseCode()} ${using(conn.getErrorStream())(fullyReadInputStreamAsString)}",
    )
    val result = using(conn.getInputStream())(readResponse)
    conn.disconnect()
    result
  }

  private[this] def fullyReadInputStreamAsString(is: InputStream): String =
    using(new ByteArrayOutputStream()) { baos =>
      drainInputStreamToOutputStream(is, baos)
      new String(baos.toByteArray(), StandardCharsets.UTF_8)
    }
}
