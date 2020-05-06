package is.hail.utils

import java.io._
import java.util._

class TracingOutputStream(
  out: OutputStream
) extends OutputStream {
  private[this] val filename = s"tracing-output-stream-${UUID.randomUUID}"
  private[this] val logfile = new FileOutputStream(filename, true)
  log.info(s"tracing to $filename")

  override def write(b: Int): Unit = {
    logfile.write(b)
    logfile.flush()
    out.write(b)
  }

  override def flush(): Unit = {
    logfile.flush()
    out.flush()
  }

  override def close(): Unit = {
    logfile.close()
    out.close()
  }
}
