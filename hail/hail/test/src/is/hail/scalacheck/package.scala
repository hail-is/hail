package is.hail

import is.hail.utils.toRichIterable

import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag
import scala.util.Random

import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Gen._
import org.scalacheck.util.Buildable

package object scalacheck
    extends DenseMatrixGenInstances with GenomicGenInstances with PTypeGenInstances
    with TypeGenInstances with GenVal {

  implicit private[scalacheck] def arbitraryFromGen[A](g: Gen[A]): Arbitrary[A] =
    Arbitrary(g)

  def distinctNonEmptyContainer[F[_], A](
    g: Gen[A]
  )(implicit
    B: Buildable[A, F[A]],
    I: F[A] => Iterable[A],
  ): Gen[F[A]] =
    nonEmptyContainerOf[F, A](g) suchThat { I(_).areDistinct() }

  def distinctContainerOfN[F[_], A](
    n: Int,
    g: Gen[A],
  )(implicit
    B: Buildable[A, F[A]],
    I: F[A] => Iterable[A],
  ): Gen[F[A]] =
    containerOfN[F, A](n, g) suchThat { I(_).areDistinct() }

  def nullable[A >: Null](gen: Gen[A], likelihood: Double = 0.01): Gen[A] =
    prob(likelihood) flatMap {
      case true => const(null)
      case false => gen
    }

  def partition[A: ClassTag: Choose](size: A, parts: Int)(implicit T: Numeric[A]): Gen[Array[A]] = {
    import T.mkOrderingOps
    assert(
      size >= T.zero,
      s"size must be greater than or equal to 0. Found $size. tn.zero=${T.zero}.",
    )

    val a: Array[A] = Array.fill[A](parts)(T.zero)
    if (a.isEmpty) const(a)
    else for {
      nSuccesses <- choose(1, parts)
      remainder <- (0 until nSuccesses - 1)
        .foldLeft(const(size)) { (gets, i) =>
          for {
            remainder <- gets
            n <- choose(T.zero, remainder)
            _ = a(i) = n
          } yield T.minus(remainder, n)
        }
      _ = a(nSuccesses - 1) = remainder
    } yield Random.shuffle(a.toFastSeq).toArray
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
