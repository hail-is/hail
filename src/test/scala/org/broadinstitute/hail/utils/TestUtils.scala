package org.broadinstitute.hail.utils

import org.scalatest.Assertions._

object TestUtils {
  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[FatalException](f)
    assert(regex.r.findFirstIn(thrown.getMessage).isDefined)

  }
}