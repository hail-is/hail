package is.hail.expr.ir.lowering

import cats.mtl.{Ask, Stateful}
import cats.{Applicative, Monad, MonadThrow}
import is.hail.backend.ExecuteContext

import scala.annotation.tailrec
import scala.language.{higherKinds, implicitConversions}

trait MonadLower[M[_]] extends MonadThrow[M] {
  def ctx: Ask[M, ExecuteContext]
  def state: Stateful[M, LoweringState]
  def lift[A](lower: Lower[A]): M[A]
}

object MonadLower {
  @inline def apply[M[_]](implicit instance: MonadLower[M]): MonadLower[M] =
    instance
}

final case class Lower[+A](run: (ExecuteContext, LoweringState) => (LoweringState, Either[Throwable, A])) {
  def runA(ctx: ExecuteContext, s: LoweringState): A =
    run(ctx, s)._2 match {
      case Right(a) => a
      case Left(t) => throw t
    }
}

object Lower extends MonadLower[Lower] {
  override def ctx: Ask[Lower, ExecuteContext] =
    new Ask[Lower, ExecuteContext] {
      override def applicative: Applicative[Lower] =
        Lower

      override def ask[E2 >: ExecuteContext]: Lower[E2] =
        Lower((ctx, s) => (s, Right(ctx.asInstanceOf[E2])))
    }

  override def state: Stateful[Lower, LoweringState] =
    new Stateful[Lower, LoweringState]{
      override def monad: Monad[Lower] =
        Lower

      override def get: Lower[LoweringState] =
        Lower((_, s) => (s, Right(s)))

      override def set(s: LoweringState): Lower[Unit] =
        Lower((_, _) => (s, Right(())))
    }

  override def lift[A](lower: Lower[A]): Lower[A] =
    lower

  override def flatMap[A, B](fa: Lower[A])(f: A => Lower[B]): Lower[B] =
    Lower { (ctx, s0) =>
      fa.run(ctx, s0) match {
        case (s1, Right(a)) => f(a).run(ctx, s1)
        case (s1, Left(t)) => (s1, Left(t))
      }
    }

  override def tailRecM[A, B](a0: A)(f: A => Lower[Either[A, B]]): Lower[B] =
    Lower { (ctx, s0) =>
      @tailrec def go(x: (LoweringState, Either[Throwable, Either[A, B]])): (LoweringState, Either[Throwable, B]) =
        x match {
          case (s, Right(Right(b))) => (s, Right(b))
          case (s, Right(Left(a))) => go(f(a).run(ctx, s))
          case (s, Left(t)) => (s, Left(t))
        }

      go((s0, Right(Left(a0))))
    }

  override def pure[A](a: A): Lower[A] =
    Lower((_, s) => (s, Right(a)))

  override def raiseError[A](e: Throwable): Lower[A] =
    Lower((_, s) => (s, Left(e)))

  override def handleErrorWith[A](fa: Lower[A])(f: Throwable => Lower[A]): Lower[A] =
    Lower { (ctx, s0) =>
      fa.run(ctx, s0) match {
        case (s1, Left(t)) => f(t).run(ctx, s1)
        case success => success
      }
    }

  implicit def monadLowerInstanceForLower: MonadLower[Lower] =
    this
}
