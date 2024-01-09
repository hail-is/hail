package is.hail.utils

import java.io._
import java.util._

class TracingInputStream(
  in: InputStream
) extends InputStream {
  private[this] val filename = s"tracing-input-stream-${UUID.randomUUID}"
  private[this] val logfile = new FileOutputStream(filename, true)
  log.info(s"tracing to $filename")

  override def read(): Int = {
    val b = in.read()
    logfile.write(b.toByte)
    b
  }

  override def close(): Unit =
    in.close()
}
