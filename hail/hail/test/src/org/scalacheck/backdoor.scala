package org.scalacheck

import org.scalacheck.Gen.{r, Parameters}
import org.scalacheck.rng.Seed

object backdoor {
  def gen[T](f: (Parameters, Seed) => (Option[T], Seed)): Gen[T] =
    Gen.gen[T]((p, s0) => f(p, s0) match { case (o, s1) => r(o, s1) })
}
