package is.hail.io.fs

import java.io._
import java.util.{concurrent => javaConcurrent}
import org.apache.commons.io.IOUtils
import org.apache.log4j.Logger
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.{Duration}
import scala.concurrent.{Future, Await, ExecutionContext}

import is.hail.services.memory_client.MemoryClient
import is.hail.services.ClientResponseException
import is.hail.utils._

trait ServiceCacheableFS extends FS {

  private[this] implicit val ec = ExecutionContext.fromExecutorService(
    javaConcurrent.Executors.newCachedThreadPool())

  private val log = Logger.getLogger(getClass.getName())

  @transient lazy val client: MemoryClient = {
    new MemoryClient("unused")
  }

  override def openCachedNoCompression(filename: String): SeekableDataInputStream = {
    client.open(filename).map(new WrappedSeekableDataInputStream(_)).getOrElse(openNoCompression(filename))
  }

  override def writeCached(filename: String)(writer: PositionedDataOutputStream => Unit) = {
    try {
      client.writeToStream(filename) { os =>
        writer(outputStreamToPositionedDataOutputStream(os))
      }
    } catch {
      case exc: Exception =>
        log.error(f"Unexpected failure $filename writing to memory. Retrying directly to GCS.", exc)
        using(createNoCompression(filename))(writer)
    }
  }

  override def createCachedNoCompression(filename: String): PositionedDataOutputStream = {
    val pis = new PipedInputStream()
    val pos = new PipedOutputStream(pis)
    val writerFuture = Future {
      client.writeToStream(filename) { os => IOUtils.copy(pis, os) }
    }
    val os: PositionedOutputStream = new OutputStream with Positioned {
      private[this] var count: Long = 0L

      override def flush(): Unit = {
        pos.flush()
        if (writerFuture.isCompleted) {
          // there might be an exception, we should eagerly retrieve that and report it
          Await.result(writerFuture, Duration.Inf)
        }
      }

      override def write(i: Int): Unit = {
        pos.write(i)
        count += 1
      }

      override def write(bytes: Array[Byte], off: Int, len: Int): Unit = {
        pos.write(bytes, off, len)
      }

      override def close(): Unit = {
        pos.close()
        Await.result(writerFuture, Duration.Inf)
      }

      def getPosition: Long = count
    }

    new WrappedPositionedDataOutputStream(os)
  }
}
