package org.broadinstitute.hail.utils

import org.slf4j.{Logger, LoggerFactory}

trait Logging {
  @transient var log_ : Logger = null

  def log: Logger = {
    if (log_ == null)
      log_ = LoggerFactory.getLogger("Hail")
    log_
  }
}
