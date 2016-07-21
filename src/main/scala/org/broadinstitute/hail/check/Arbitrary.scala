package org.broadinstitute.hail.check

import scala.collection.generic.CanBuildFrom

object Arbitrary {
  def apply[T](arbitrary: Gen[T]): Arbitrary[T] =
    new Arbitrary(arbitrary)

  implicit def arbBoolean: Arbitrary[Boolean] = new Arbitrary(
    Gen.oneOf(true, false))

  implicit def arbByte: Arbitrary[Byte] = new Arbitrary(Gen.arbByte)

  implicit def arbInt: Arbitrary[Int] = new Arbitrary(
    Gen.oneOfGen(Gen.oneOf(Int.MinValue, -1, 0, 1, Int.MaxValue),
      Gen.choose(-100, 100),
      Gen.arbInt))

  implicit def arbLong: Arbitrary[Long] = new Arbitrary(
    Gen.oneOfGen(Gen.oneOf(Long.MinValue, -1L, 0L, 1L, Long.MaxValue),
      Gen.choose(-100, 100),
      Gen.arbLong))

  implicit def arbDouble: Arbitrary[Double] = new Arbitrary(
    Gen.oneOfGen(Gen.oneOf(Double.MinValue, -1.0, 0.0, Double.MinPositiveValue, 1.0, Double.MaxValue),
      Gen.choose(-100.0, 100.0),
      Gen.arbDouble))

  implicit def arbString: Arbitrary[String] = new Arbitrary(Gen.arbString)

  implicit def arbBuildableOf[C, T](implicit a: Arbitrary[T], cbf: CanBuildFrom[Nothing, T, C]): Arbitrary[C] =
    Arbitrary(Gen.buildableOf[C, T](a.arbitrary))

  def arbitrary[T](implicit arb: Arbitrary[T]): Gen[T] = arb.arbitrary
}

class Arbitrary[T](val arbitrary: Gen[T])
