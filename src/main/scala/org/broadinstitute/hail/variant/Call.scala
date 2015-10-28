package org.broadinstitute.hail.variant

import org.broadinstitute.hail.Utils._

case class Call(gt: Int, gq: Int, pl: (Int, Int, Int)) {
  require(gt >= 0 && gt <= 2)
  require(gq >= 0 && gq <= 99)
  require(pl.at(gt + 1) == 0)
}
