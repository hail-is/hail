package org.broadinstitute.k3.variant

case class Call(gt: Int, gq: Int, pl: (Int, Int, Int)) {
  require(gt >= 0 && gt <= 2)
  require(gq >= 0 && gq <= 99)
  // FIXME pl(gt) == 0
}
