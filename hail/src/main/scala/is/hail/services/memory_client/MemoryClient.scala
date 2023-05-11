package is.hail.services.memory_client

import java.io._

import is.hail.HailContext
import is.hail.io.fs.{Seekable, Positioned}
import is.hail.services.{ClientResponseException, DeployConfig, Requester, Tokens}
import org.apache.http.client.methods._
import org.apache.http.entity._
import org.apache.http._
import org.apache.http.client.utils.URIBuilder
import org.apache.log4j.{LogManager, Logger}
import org.json4s.JValue

object MemoryClient {
  private val log = Logger.getLogger(getClass.getName())
}

class MemoryClient(val deployConfig: DeployConfig, requester: Requester) {

  def this(credentialsPath: String) = this(DeployConfig.get, Requester.fromCredentialsFile(credentialsPath))

  import MemoryClient.log
  import requester.request

  private[this] val baseUrl = deployConfig.baseUrl("memory")

  def open(path: String): Option[MemorySeekableInputStream] =
    try {
      Some(new MemorySeekableInputStream(requester, s"$baseUrl/api/v1alpha/objects", path))
    } catch { case e: ClientResponseException if e.status == 404 =>
      None
    }

  def write(path: String, data: Array[Byte]): Unit = {
    val uri = new URIBuilder(s"$baseUrl/api/v1alpha/objects").addParameter("q", path)
    val req = new HttpPost(uri.build())
    requester.request(req, new ByteArrayEntity(data))
  }

  def writeToStream(
    path: String,
    _isRepeatable: Boolean = false,
    _contentType: String = "application/octet-stream",
    _contentLength: Long = -1,
    _isChunked: Boolean = true
  )(
    writer: OutputStream => Unit
  ): Unit = {
    val uri = new URIBuilder(s"$baseUrl/api/v1alpha/objects").addParameter("q", path)
    val req = new HttpPost(uri.build())
    val entity = new AbstractHttpEntity {
      def isRepeatable(): Boolean = _isRepeatable
      def writeTo(os: OutputStream): Unit = writer(os)
      def getContentLength() = _contentLength
      def getContent() = throw new UnsupportedOperationException()
      def isStreaming() = true
    }
    entity.setContentType(_contentType)
    entity.setChunked(_isChunked)
    requester.request(req, entity)
  }
}

class MemorySeekableInputStream(requester: Requester, objectEndpoint: String, fileURI: String) extends InputStream with Seekable {
  private[this] val uri = new URIBuilder(objectEndpoint)
    .addParameter("q", fileURI)

  private[this] val req = new HttpGet(uri.build())

  private[this] val _bytes = requester.requestAsByteStream(req)
  private[this] var _pos: Int = 0
  private[this] val _len: Int = _bytes.length

  override def read(): Int = {
    if (_pos >= _len) -1 else {
      _pos += 1
      _bytes(_pos - 1).toInt & 0xff
    }
  }

  override def read(bytes: Array[Byte], off: Int, len: Int): Int = {
    if (off < 0 || len < 0 || len > bytes.length - off)
      throw new IndexOutOfBoundsException()
    if (len == 0)
      0
    else if (_pos >= _len)
      -1
    else {
      val n_read: Int = java.lang.Math.min(_len - _pos, len)
      System.arraycopy(_bytes, _pos, bytes, off, n_read)
      _pos += n_read
      n_read
    }
  }

  override def skip(n: Long): Long = {
    if (n <= 0L)
      0L
    else {
      if (_pos + n >= _len) {
        val r = _len - _pos
        _pos = _len
        r
      } else {
        _pos = _pos + n.toInt
        n
      }
    }
  }

  override def close(): Unit = {}

  def getPosition: Long = _pos

  def seek(pos: Long): Unit =
    if (pos < 0 || pos > _len)
      throw new IOException(s"Cannot seek to position $pos")
    else _pos = pos.toInt
}
