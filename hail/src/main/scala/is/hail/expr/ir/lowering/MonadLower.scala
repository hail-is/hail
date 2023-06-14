package is.hail.expr.ir.lowering

import cats.mtl.Stateful
import cats.{Applicative, Monad}
import is.hail.backend.{ExecuteContext, MonadExecute}

import scala.annotation.tailrec
import scala.language.{higherKinds, implicitConversions}

trait MonadLower[M[_]]
  extends MonadExecute[M]
    with Stateful[M, LoweringState] {
  def liftLower[A](lower: Lower[A]): M[A]
}

object MonadLower {
  @inline def apply[M[_]: MonadLower]: MonadLower[M] =
    implicitly
}



final case class Lower[+A](run: (ExecuteContext, LoweringState) => (LoweringState, Either[Throwable, A])) {
  def runA(ctx: ExecuteContext, s: LoweringState): A =
    run(ctx, s)._2.fold(throw _, identity)
}

object Lower extends MonadLower[Lower] {
  override def applicative: Applicative[Lower] =
    Lower
  override def monad: Monad[Lower] =
    Lower

  override def ask[E2 >: ExecuteContext]: Lower[E2] =
    Lower((ctx, s) => (s, Right(ctx.asInstanceOf[E2])))

  override def local[A](fa: Lower[A])(f: ExecuteContext => ExecuteContext): Lower[A] =
    Lower((ctx, s) => fa.run(f(ctx), s))

  override def get: Lower[LoweringState] =
    Lower((_, s) => (s, Right(s)))

  override def set(s: LoweringState): Lower[Unit] =
    Lower((_, _) => (s, Right(())))

  override def liftLower[A](lower: Lower[A]): Lower[A] =
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

  override def raiseError[A](e: Throwable): Lower[A] = {
    e.fillInStackTrace()
    Lower((_, s) => (s, Left(e)))
  }

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
