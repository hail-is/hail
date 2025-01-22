package is.hail.check

import org.scalacheck.Gen
import org.scalacheck.util.Buildable

import scala.language.higherKinds

object GenMore {
  def distinctNonEmptyContainer[F[_], A](
    g: Gen[A]
  )(implicit
    evb: Buildable[A, F[A]],
    evt: F[A] => Traversable[A],
  ): Gen[F[A]] =
    Gen.nonEmptyContainerOf[F, A](g) suchThat { xs => xs.size == xs.toSet.size }

  def distinctContainerOfN[F[_], A](
    n: Int,
    g: Gen[A],
  )(implicit
    evb: Buildable[A, F[A]],
    evt: F[A] => Traversable[A],
  ): Gen[F[A]] =
    Gen.containerOfN[F, A](n, g) suchThat { xs => xs.size == xs.toSet.size }
}
