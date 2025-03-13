package is.hail.check

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds

object Arbitrary {
  def apply[T](arbitrary: Gen[T]): Arbitrary[T] =
    new Arbitrary(arbitrary)

  implicit def arbBoolean: Arbitrary[Boolean] = new Arbitrary(
    Gen.oneOf(true, false)
  )

  implicit def arbByte: Arbitrary[Byte] = new Arbitrary(Gen.oneOfGen(
    Gen.oneOf(Byte.MinValue, -1, 0, 1, Byte.MaxValue),
    Gen(p => p.rng.getRandomGenerator.nextInt().toByte),
  ))

  implicit def arbInt: Arbitrary[Int] = new Arbitrary(
    Gen.oneOfGen(
      Gen.oneOf(Int.MinValue, -1, 0, 1, Int.MaxValue),
      Gen.choose(-100, 100),
      Gen(p => p.rng.getRandomGenerator.nextInt()),
    )
  )

  implicit def arbLong: Arbitrary[Long] = new Arbitrary(
    Gen.oneOfGen(
      Gen.oneOf(Long.MinValue, -1L, 0L, 1L, Long.MaxValue),
      Gen.choose(-100, 100),
      Gen(p => p.rng.getRandomGenerator.nextLong()),
    )
  )

  implicit def arbFloat: Arbitrary[Float] = new Arbitrary(
    Gen.oneOfGen(
      Gen.oneOf(
        Float.MinValue,
        -1.0f,
        -Float.MinPositiveValue,
        0.0f,
        Float.MinPositiveValue,
        1.0f,
        Float.MaxValue,
      ),
      Gen.choose(-100.0f, 100.0f),
      Gen(p => p.rng.nextUniform(Float.MinValue, Float.MaxValue, true).toFloat),
    )
  )

  implicit def arbDouble: Arbitrary[Double] = new Arbitrary(
    Gen.oneOfGen(
      Gen.oneOf(
        Double.MinValue,
        -1.0,
        -Double.MinPositiveValue,
        0.0,
        Double.MinPositiveValue,
        1.0,
        Double.MaxValue,
      ),
      Gen.choose(-100.0, 100.0),
      Gen(p => p.rng.nextUniform(Double.MinValue, Double.MaxValue, true)),
    )
  )

  implicit def arbString: Arbitrary[String] = new Arbitrary(Gen.frequency(
    (1, Gen.const("")),
    (
      10,
      Gen { (p: Parameters) =>
        val s = p.rng.getRandomGenerator.nextInt(12)
        val b = new StringBuilder()
        for (i <- 0 until s)
          b += Gen.randomOneOf(p.rng, Gen.printableChars)
        b.result()
      },
    ),
  ))

  implicit def arbBuildableOf[C[_], T](
    implicit a: Arbitrary[T],
    cbf: CanBuildFrom[Nothing, T, C[T]],
  ): Arbitrary[C[T]] =
    Arbitrary(Gen.buildableOf(a.arbitrary))

  def arbitrary[T](implicit arb: Arbitrary[T]): Gen[T] = arb.arbitrary
}

class Arbitrary[T](val arbitrary: Gen[T])
