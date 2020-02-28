package is.hail.shuffler

import is.hail.HailContext
import is.hail.utils._
import is.hail.expr.ir.ExecuteContext
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream, InputStream, InputStreamReader, OutputStream }
import scala.reflect.ClassTag

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
    }
    assert(rootUrl != null)
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

  private[this] val me = getWorkers()(0)

  def write(
    writer: OutputStream => Unit
  ): Key = {
    HTTPClient.post(sessionUrl, 0, writer, { in =>
      val baos = new ByteArrayOutputStream()
      drainInputStreamToOutputStream(in, baos)
      val bytes = baos.toByteArray()
      val JArray(List(JString(s), JInt(fileId), JInt(pos), JInt(n))) =
        JsonMethods.parse(new InputStreamReader(new ByteArrayInputStream(bytes)))
      val x = (s, fileId.toInt, pos.toInt, n.toInt)
      log.info(s"wrote ${x}")
      x
    })
  }

  def read[T](
    key: Key,
    reader: InputStream => T
  ): T =
    HTTPClient.post(s"${rootUrl.replace(me, key._1)}/s/${id}/get", 0, { out =>
      Serialization.write(Array(key._1, key._2, key._3, key._4), out)
    }, reader)

  def readMany[T: ClassTag](
    keys: Array[Key],
    reader: InputStream => T
  ): Array[T] = {
    assert(keys.length > 0)
    log.info(s"fetching ${keys.length} keys from ${keys(0)._1}: ${keys.mkString(",")}")
    var i = 0
    HTTPClient.post(s"${rootUrl.replace(me, keys(0)._1)}/s/${id}/getmany", 0, { out =>
      Serialization.write(keys.map(key => Array(key._1, key._2, key._3, key._4)), out)
    }, decode(_, reader))
  }

  private[this] def decode[T: ClassTag](
    in: InputStream,
    reader: InputStream => T
  ): Array[T] = {
    def readInt: Int = {
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
    val slice = new SlicedInputStream(in)
    val ab = new ArrayBuilder[T]()
    var n = readInt
    while (n >= 0) {
        assert(n >= 0)
        slice.startSlice(n)
        ab += reader(slice)
        n = readInt
    }
    ab.result()
  }

  def delete(): Unit = HTTPClient.delete(sessionUrl)

  def getWorkers(): Seq[String] =
    HTTPClient.get(s"${rootUrl}/w", { in =>
      val JArray(l) = JsonMethods.parse(new InputStreamReader(in))
      l.map { case JString(s) => s }
    })
}
