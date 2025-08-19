package scala
package util

trait ChainingSyntax {
  @inline implicit final def scalaUtilChainingOps[A](a: A): ChainingOps[A] = new ChainingOps(a)
}

final class ChainingOps[A](private val self: A) extends AnyVal {
  def tap[U](f: A => U): A = {
    f(self)
    self
  }

  def pipe[B](f: A => B): B = f(self)
}
