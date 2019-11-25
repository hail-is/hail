package is.hail.shuffler

import is.hail.HailContext
import is.hail.utils._
import is.hail.expr.ir.ExecuteContext
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream, InputStream, InputStreamReader, OutputStream }

import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

object BufferClient {
  type Key = (String, Int, Int, Int)

  private implicit val f = new DefaultFormats() {}

  def create(
    executeContext: ExecuteContext,
    maybeRootUrl: String = null
  ): BufferClient = {
    var rootUrl = maybeRootUrl
    if (rootUrl == null) {
      rootUrl = HailContext.get.flags.get("buffer_service_url")
      if (rootUrl == null) {
        rootUrl = "http://localhost:5000"
      }
    }
    val client = new BufferClient(
      rootUrl,
      HTTPClient.post(s"${rootUrl}/s",
        0, out => (), in => Serialization.read[Int](new InputStreamReader(in))))
    val url = client.sessionUrl
    Runtime.getRuntime.addShutdownHook(new Thread(new Runnable() {
      override def run(): Unit = {
        log.info(s"shutdown hook buffer ${client.id}")
        HTTPClient.delete(url)
      }
    }))
    executeContext.addOnExit { () =>
      HTTPClient.delete(url)
    }
    client
  }
}

class BufferClient (
  val rootUrl: String,
  val id: Int
) extends Serializable {
  import BufferClient._

  val sessionUrl = s"${rootUrl}/s/${id}"

  def write(
    writer: OutputStream => Unit
  ): Key = HTTPClient.post(sessionUrl, 0, writer, { in =>
    val baos = new ByteArrayOutputStream()
    drainInputStreamToOutputStream(in, baos)
    val bytes = baos.toByteArray()
    val JArray(List(JString(s), JInt(fileId), JInt(pos), JInt(n))) =
      JsonMethods.parse(new InputStreamReader(new ByteArrayInputStream(bytes)))
    (s, fileId.toInt, pos.toInt, n.toInt)
  })

  def read[T](
    key: Key,
    reader: InputStream => T
  ): T =
    HTTPClient.post(s"http://${key._1}/s/${id}/get", 0, { out =>
      Serialization.write(Array(key._1, key._2, key._3, key._4), out)
    }, reader)

  def readMany[T](
    keys: Array[Key],
    reader: InputStream => T
  ): Iterator[T] = {
    assert(keys.length > 0)
    val server = keys(0)._1
    HTTPClient.post(s"http://${server}/s/${id}/getmany", 0, { out =>
      Serialization.write(keys.map(key => Array(key._1, key._2, key._3, key._4)), out)
    }, decoder(_, reader))
  }

  private[this] def decoder[T](
    in: InputStream,
    reader: InputStream => T
  ): Iterator[T] = new Iterator[T] {
    private[this] def readInt: Int = {
      var out = 0
      var i = in.read()
      if (i == -1)
        return i
      out |= (i & 0xff)
      i = in.read()
      if (i == -1)
        return i
      out |= (i & 0xff) << 8
      i = in.read()
      if (i == -1)
        return i
      out |= (i & 0xff) << 16
      i = in.read()
      if (i == -1)
        return i
      out | (i & 0xff) << 24
    }
    private[this] val slice = new SlicedInputStream(in)
    private[this] var n = readInt
    def hasNext: Boolean = n < 0

    def next(): T = {
      assert(n >= 0)
      slice.startSlice(n)
      n = readInt
      reader(slice)
    }
  }

  def delete(): Unit = HTTPClient.delete(sessionUrl)

  def getWorkers(): Seq[String] =
    HTTPClient.get(s"${rootUrl}/w", { in =>
      val JArray(l) = JsonMethods.parse(new InputStreamReader(in))
      l.map { case JString(s) => s }
    })
}
