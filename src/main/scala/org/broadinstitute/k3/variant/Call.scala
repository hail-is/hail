package org.broadinstitute.k3.variant

import org.broadinstitute.k3.Utils._

case class Call(gt: Int, gq: Int, pl: (Int, Int, Int)) {
  require(gt >= 0 && gt <= 2)
  require(gq >= 0 && gq <= 99)
  require(pl.at(gt) == 0)
}
