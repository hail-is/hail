package is.hail.io.fs

import java.io.{InputStream, OutputStream}
import org.apache.log4j.Logger
import scala.collection.mutable.ArrayBuffer

import is.hail.services.memory_client.MemoryClient
import is.hail.services.ClientResponseException
import is.hail.utils._

trait ServiceCacheableFS extends FS {

  def sessionID: String

  private val log = Logger.getLogger(getClass.getName())

  @transient lazy val client: MemoryClient = {
    if (sessionID != null)
      MemoryClient.fromSessionID(sessionID)
    else MemoryClient.get
  }

  def openCachedNoCompression(filename: String): SeekableDataInputStream = {
    client.open(filename).map(new WrappedSeekableDataInputStream(_)).getOrElse(openNoCompression(filename))
  }

  def createCachedNoCompression(filename: String): PositionedDataOutputStream = {
    val os: PositionedOutputStream = new OutputStream with Positioned {
      private[this] var closed: Boolean = false
      private[this] val bb: ArrayBuffer[Byte] = new ArrayBuffer()

      override def flush(): Unit = {}

      override def write(i: Int): Unit = {
        bb += i.toByte
      }

      override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
        bb ++= bytes.slice(off, off + len)
      }

      override def close(): Unit = {
        if (!closed) {
          val data = bb.toArray
          try {
            client.write(filename, data)
          } catch { case e: Exception =>
            log.error(s"Failed to upload $filename to memory service, falling back to GCS")
            using(createNoCompression(filename)) { os =>
              os.write(data)
            }
          }
          closed = true
        }
      }

      def getPosition: Long = bb.length
    }

    new WrappedPositionedDataOutputStream(os)
  }
}
