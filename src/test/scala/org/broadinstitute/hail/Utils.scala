package org.broadinstitute.hail

import org.broadinstitute.hail.utils.FatalException

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[FatalException](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected fatal exception with pattern `$regex'
           |  Found: ${thrown.getMessage}""".stripMargin)
    assert(p)
  }
}
