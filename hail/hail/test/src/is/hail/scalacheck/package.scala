package is.hail

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag
import scala.util.Random

import java.lang.Math.max

import org.apache.commons.math3.distribution.{BetaDistribution, GammaDistribution}
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen._
import org.scalacheck.util.Buildable

package object scalacheck
    extends ArbitraryDenseMatrixInstances with ArbitraryGenomicInstances
    with ArbitraryPTypeInstances with ArbitraryTypeInstances with GenVal {

  implicit private[scalacheck] def arbitraryFromGen[A](g: Gen[A]): Arbitrary[A] =
    Arbitrary(g)

  implicit def buildableArray[A: ClassTag]: Buildable[A, Array[A]] =
    new Buildable[A, Array[A]] {
      override def builder: mutable.Builder[A, Array[A]] =
        Array.newBuilder
    }

  def distinctNonEmptyContainer[F[_], A](
    g: Gen[A],
    maxCollisions: Int = 1000,
  )(implicit
    B: Buildable[A, F[A]]
  ): Gen[F[A]] =
    for {
      k <- size
      n <- choose(1, max(1, k))
      c <- distinctContainerOfN[F, A](n, g, maxCollisions)
    } yield c

  def distinctContainerOfN[F[_], A](
    n: Int,
    g: Gen[A],
    maxCollisions: Int = 1000,
  )(implicit
    B: Buildable[A, F[A]]
  ): Gen[F[A]] =
    tailRecM((mutable.Set.empty[A], 0)) {
      case (elems, collisions) =>
        if (elems.size == n) const(Right(B.fromIterable(elems)))
        else if (collisions >= maxCollisions) throw MaxCollisionsExceeded(elems, n, maxCollisions)
        else g map { e =>
          if (elems.contains(e)) Left((elems, collisions + 1))
          else Left((elems += e, collisions))
        }
    }

  private[scalacheck] def MaxCollisionsExceeded(s: Traversable[_], size: Int, collisions: Int) =
    new RuntimeException(
      f"Failed to generate $size number of elements after $collisions collisions." +
        f"Elements are: ${s.mkString("[", ", ", "]")}"
    )

  def nullable[A >: Null](gen: Gen[A], likelihood: Double = 0.01): Gen[A] =
    prob(likelihood) flatMap {
      case true => const(null)
      case false => gen
    }

  def smaller[A: Arbitrary]: Gen[A] =
    beta(1, 3) flatMap { p => sized(s => resize((p * s).toInt, arbitrary[A])) }

  def limit[A: Arbitrary](len: Int): Gen[A] =
    sized(s => resize(Math.min(len, s), arbitrary[A]))

  private[scalacheck] lazy val rngGen: Gen[RandomGenerator] =
    delay(long map { new Well19937c(_) })

  def beta(alpha: Double, beta: Double): Gen[Double] =
    rngGen map { new BetaDistribution(_, alpha, beta).sample() }

  def gamma(shape: Double, scale: Double): Gen[Double] =
    rngGen map { new GammaDistribution(_, shape, scale).sample() }

  def dirichlet(alpha: Array[Double]): Gen[Array[Double]] =
    sequence(alpha.map(gamma(_, 1))) map { samples =>
      val sum = samples.sum; samples.map(_ / sum)
    }

  def partition[A: ClassTag: Choose](sum: A, parts: Int)(implicit T: Numeric[A]): Gen[Array[A]] =
    choose(1, parts) flatMap { nSuccesses =>
      val Zero = T.zero
      tailRecM((new ArrayBuffer[A](parts), 0, sum)) {
        case (buff, idx, Zero) =>
          for (_ <- idx until parts) buff += Zero
          const(Right((Random.shuffle(buff), parts, Zero)))

        case (buff, idx, remainder) if idx == nSuccesses - 1 =>
          const(Left((buff += remainder, idx + 1, Zero)))

        case (buff, idx, remainder) if idx < nSuccesses - 1 =>
          choose(Zero, remainder) map { n => Left((buff += n, idx + 1, T.minus(remainder, n))) }
      } map { _._1.toArray }
    }

  def liftA2[A, B, C](f: (A, B) => C, genA: Gen[A], genB: Gen[B]): Gen[C] =
    for {
      a <- genA
      b <- genB
    } yield f(a, b)

  def liftA3[A, B, C, D](f: (A, B, C) => D, genA: Gen[A], genB: Gen[B], genC: Gen[C]): Gen[D] =
    for {
      a <- genA
      b <- genB
      c <- genC
    } yield f(a, b, c)

  implicit class ApplicativeGenOps[A, B](genF: Gen[A => B]) {
    def ap[C <: A](gen: Gen[C]): Gen[B] =
      genF.flatMap(gen.map)
  }
}
