package org.broadinstitute.hail.utils

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

  def warn(msg: String) {
    log.warn(msg)
    System.err.println("hail: warning: " + msg)
  }

  def error(msg: String) {
    log.error(msg)
    System.err.println("hail: error: " + msg)
  }
}
