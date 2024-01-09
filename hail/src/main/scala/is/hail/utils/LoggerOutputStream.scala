package is.hail.utils

import java.io.{ByteArrayOutputStream, OutputStream}
import java.nio.charset.StandardCharsets

import org.apache.log4j.{Level, Logger}

class LoggerOutputStream(logger: Logger, level: Level) extends OutputStream {
  private val buffer = new ByteArrayOutputStream()

  override def write(b: Int) {
    buffer.write(b)
    if (b == '\n') {
      val line = buffer.toString(StandardCharsets.UTF_8.name())
      level match {
        case Level.TRACE => logger.trace(line)
        case Level.DEBUG => logger.debug(line)
        case Level.INFO => logger.info(line)
        case Level.WARN => logger.warn(line)
        case Level.ERROR => logger.error(line)
      }
      buffer.reset()
    }
  }
}
