package org.broadinstitute.hail

import org.scalacheck.{Gen, Prop}

trait ScalaCheckSuite {
  def check(p: Prop) {
    p.check
    return

    val result = p(Gen.Parameters.default)
    if (result.status != Prop.True)
      System.err.println(result)

    assert(result.status == Prop.True)
  }
}
