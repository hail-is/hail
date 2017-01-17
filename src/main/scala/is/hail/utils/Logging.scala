package is.hail.utils

import org.slf4j.{Logger, LoggerFactory}

trait Logging {
  @transient var log_ : Logger = _

  def log: Logger = {
    if (log_ == null)
      log_ = LoggerFactory.getLogger("Hail")
    log_
  }

  def info(msg: String) {
    log.info(msg)
    System.err.println("hail: info: " + msg)
  }

  def info(msg: String, t: Truncatable) {
    val (screen, logged) = t.strings

    log.info(format(msg, logged))
    System.err.println("hail: info: " + format(msg, screen))
  }

  def warn(msg: String) {
    log.warn(msg)
    System.err.println("hail: warning: " + msg)
  }

  def warn(msg: String, t: Truncatable) {
    val (screen, logged) = t.strings

    log.warn(format(msg, logged))
    System.err.println("hail: warning: " + format(msg, screen))
  }
}
