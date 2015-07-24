package org.broadinstitute.k3.variant

case class Call(GT: Int, GQ: Int, PL: (Int, Int, Int)) {
  require(GT >= 0 && GT <= 2)
  require(GQ >= 0 && GQ <= 99)
}
