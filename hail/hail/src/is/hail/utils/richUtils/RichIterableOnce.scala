package is.hail.utils.richUtils

import scala.collection.compat.{BuildFrom, IterableOnce}

class RichIterableOnce[F[+X] <: IterableOnce[X], A](val xs: F[A]) extends AnyVal {

  def partitionEither[L, R](
    implicit
    EV: A <:< Either[L, R],
    BuildLefts: BuildFrom[F[A], L, F[L]],
    BuildRights: BuildFrom[F[A], R, F[R]],
  ): (F[L], F[R]) = {
    val ls = BuildLefts.newBuilder(xs)
    val rs = BuildRights.newBuilder(xs)
    for (x <- xs) EV(x).fold[Unit](ls += _, rs += _)
    (ls.result(), rs.result())
  }
}
