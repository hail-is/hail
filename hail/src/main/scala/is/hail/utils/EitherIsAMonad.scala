package is.hail.utils

import scala.language.implicitConversions

object EitherIsAMonad {
  implicit def eitherIsAMonad[A, B](eab: Either[A, B]): EitherOps[A, B] = new EitherOps(eab)
}

final class EitherOps[A, B](val eab: Either[A, B]) extends AnyVal {
  def foreach(f: B => Unit): Unit = eab match {
    case Left(_) => ()
    case Right(b) => f(b)
  }

  def getOrElse(default: => B): B = eab match {
    case Left(_) => default
    case Right(b) => b
  }

  def valueOr(f: A => B): B = eab match {
    case Left(a) => f(a)
    case Right(b) => b
  }

  def toOption: Option[B] = eab match {
    case Left(_) => None
    case Right(b) => Some(b)
  }

  def map[C](f: B => C): Either[A, C] = eab match {
    case l @ Left(_) => l.asInstanceOf[Either[A, C]]
    case Right(b) => Right(f(b))
  }

  def flatMap[D](f: B => Either[A, D]): Either[A, D] = eab match {
    case l @ Left(_) => l.asInstanceOf[Either[A, D]]
    case Right(b) => f(b)
  }

}
