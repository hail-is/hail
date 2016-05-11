package org.broadinstitute.hail

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[FatalException](f)
    assert(regex.r.findFirstIn(thrown.getMessage).isDefined)
  }
}
