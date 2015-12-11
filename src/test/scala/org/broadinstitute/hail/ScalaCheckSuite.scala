package org.broadinstitute.hail

import org.scalacheck.{Gen, Prop}

trait ScalaCheckSuite {
  def check(p: Prop) {
    assert(p(Gen.Parameters.default).status == Prop.True)
  }
}
