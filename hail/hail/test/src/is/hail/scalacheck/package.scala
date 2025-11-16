package is.hail

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import org.apache.commons.math3.distribution.{
  BetaDistribution, BinomialDistribution, GammaDistribution,
}
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

  def disjointSetOfN[A](n: Int, collection: Iterable[A]): Gen[Set[A]] =
    if (n == 0)
      const(Set.empty[A])
    else if (n > collection.size)
      throw new RuntimeException("Requested number distinct of elements exceeds collection size")
    else
      distinctContainerOfN[Set, A](n, oneOf(collection))

  def distinctContainerOf[F[_], A](
    g: Gen[A],
    maxCollisions: Int = 1000,
  )(implicit
    B: Buildable[A, F[A]]
  ): Gen[F[A]] =
    sized(distinctContainerOfN(_, g, maxCollisions))

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

  private[scalacheck] def MaxCollisionsExceeded(s: Iterable[_], size: Int, collisions: Int) =
    new RuntimeException(
      f"Failed to generate $size distinct elements after $collisions collisions." +
        f"Elements are: ${s.mkString("[", ", ", "]")}"
    )

  def nullable[A >: Null](gen: Gen[A], likelihood: Double = 0.05): Gen[A] =
    prob(likelihood) flatMap {
      case true => const(null)
      case false => gen
    }

  def scale[A](factor: Double, gen: Gen[A]): Gen[A] =
    sized(s => resize((factor * s).ceil.toInt, gen))

  def smaller[A: Arbitrary]: Gen[A] =
    beta(1, 3) flatMap { ratio => scale[A](ratio, arbitrary[A]) }

  def atLeast[A](n: Int)(f: Int => Gen[A]): Gen[A] =
    sized(x => choose(n, math.max(x, n)) flatMap { len => resize(len, f(len)) })

  def atMost[A](n: Int)(f: Int => Gen[A]): Gen[A] =
    sized(x => choose(0, math.min(x, n)) flatMap { len => resize(len, f(len)) })

  private[scalacheck] def rngGen: Gen[RandomGenerator] =
    long map { new Well19937c(_) }

  def beta(alpha: Double, beta: Double): Gen[Double] =
    rngGen map { new BetaDistribution(_, alpha, beta).sample() }

  def binomial(trials: Int, prob: Double): Gen[Int] =
    rngGen map { new BinomialDistribution(_, trials, prob).sample() }

  def gamma(shape: Double, scale: Double): Gen[Double] =
    rngGen map { new GammaDistribution(_, shape, scale).sample() }

  def dirichlet(alpha: Array[Double]): Gen[Array[Double]] =
    sequence(alpha.map(gamma(_, 1))) map { samples =>
      val sum = samples.sum; samples.map(_ / sum)
    }

  def multinomial(trials: Int, probs: Array[Double]): Gen[Array[Int]] =
    tailRecM((new ArrayBuffer[Int](probs.length), trials, 1.0d, probs.toList)) {
      case (buff, _, _, Nil) =>
        const(Right(buff.toArray))
      case (buff, rem, _, _ :: Nil) =>
        const(Left((buff += rem, 0, 0, List.empty[Double])))
      case (buff, rem, rho, p :: ps) =>
        binomial(rem, p / rho) map { outcome =>
          Left((buff += outcome, rem - outcome, rho - p, ps))
        }
    }

  def partition(n: Int): Gen[Array[Int]] =
    sized(partition(n, _))

  def partition(n: Int, size: Int): Gen[Array[Int]] =
    dirichlet(Array.fill(n)(1d)) flatMap { multinomial(size, _) }

  def distribute[A: ClassTag](n: Int, gen: Gen[A]): Gen[Array[A]] =
    partition(n) flatMap { sizes => sequence(sizes map { resize(_, gen) }) }

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
