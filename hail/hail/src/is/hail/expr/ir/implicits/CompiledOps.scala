package is.hail.expr.ir.implicits

import is.hail.expr.ir.Compiled

private[implicits] class CompiledOps[A](val fa: Compiled[A]) extends AnyVal {
  def map[B](f: A => B): Compiled[B] =
    (hcl, fs, htc, r) =>
      f(fa(hcl, fs, htc, r))

  def flatMap[B](f: A => Compiled[B]): Compiled[B] =
    (hcl, fs, htc, r) =>
      f(fa(hcl, fs, htc, r))(hcl, fs, htc, r)
}
