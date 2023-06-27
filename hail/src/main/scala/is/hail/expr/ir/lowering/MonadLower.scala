package is.hail.expr.ir.lowering

import cats.mtl.{Local, Stateful}
import cats.{Applicative, Monad, MonadThrow, StackSafeMonad}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.Lower.{>=>, Return}

import scala.annotation.tailrec
import scala.language.{higherKinds, implicitConversions}

trait MonadLower[M[_]]
  extends MonadThrow[M]
    with Local[M, ExecuteContext]
    with Stateful[M, LoweringState] {
  def liftLower[A](lower: Lower[A]): M[A]
}


object MonadLower {
  @inline def apply[M[_]: MonadLower]: MonadLower[M] =
    implicitly
}


sealed trait Lower[+A] {
  final def run(ctx: ExecuteContext, s: LoweringState): (LoweringState, Either[Throwable, A]) = {
    @tailrec def go[B](head: Lower[B], tail: Lower.Chain[B, A])(s0: LoweringState)
    : (LoweringState, Either[Throwable, A]) =
      head match {
       case leaf: Lower.Leaf[Any] =>
         val (s1, b) = leaf match {
           case Lower.Pure(x) =>
             (s0, x)

           case Lower.Ask(ev) =>
             (s0, ev(ctx))

           case Lower.Get =>
             (s0, s0)

           case Lower.Set(s) =>
             (s, ())
         }

         tail match {
           case Return(refl) => (s1, Right(refl(b.asInstanceOf[B])))
           case f >=> g => go(f(b.asInstanceOf[B]), g)(s1)
         }

       case Lower.Apply(f) =>
         f(ctx, s) match {
           case (s1, Right(b)) => tail match {
             case Return(refl) => (s1, Right(refl(b)))
             case f >=> g => go(f(b), g)(s1)
           }
           case (s1, Left(t)) => (s1, Left(t))
         }

        case f: Lower.FlatMap[B] =>
          f.init match {
            case l: Lower[f.Init] => go(l, f.bind >=> tail)(s0)
          }

       case Lower.Fail(t) =>
         (s0, Left(t))

        case Lower.Catch(fb, onError) =>
          fb.run(ctx, s0) match {
            case (s1, Right(a)) => go(Lower.Pure(a), tail)(s1)
            case (s1, Left(t)) => go(onError(t), tail)(s1)
          }
      }

    go(this, Return(implicitly[A <:< A]))(s)
  }

  def runA(ctx: ExecuteContext, s: LoweringState): A =
    run(ctx, s)._2.fold(throw _, identity)

  def runS(ctx: ExecuteContext, s0: LoweringState): LoweringState = {
    val (s1, res) = run(ctx, s0)
    res.fold(throw _, _ => s1)
  }
}

object Lower extends LowerInstances {
  def apply[A](f: (ExecuteContext, LoweringState) => (LoweringState, Either[Throwable, A])): Lower[A] =
    Apply(f)


  sealed trait Leaf[+A] extends Lower[A]
  final case class Ask[A](ev: ExecuteContext <:< A) extends Leaf[A]
  final case class Pure[+A](value: A) extends Leaf[A]
  final case object Get extends Leaf[LoweringState]
  final case class Set(state: LoweringState) extends Leaf[Unit]


  final case class Apply[+A](f: (ExecuteContext, LoweringState) => (LoweringState, Either[Throwable, A]))
    extends Lower[A]

  abstract class FlatMap[A] extends Lower[A] {
    type Init
    val init: Lower[Init]
    val bind: Init => Lower[A]
  }


  final case class Fail[+A](t: Throwable) extends Lower[A]
  final case class Catch[+A](fa: Lower[A], handle: Throwable => Lower[A]) extends Lower[A]


  private sealed abstract class Chain[A, B]
  private final case class Return[A, B](refl: A => B) extends Chain[A, B]
  private final case class >=>[A, B, C](f: A => Lower[B], chain: Chain[B, C]) extends Chain[A, C]

  private implicit class ChainOps[A, B](val f: A => Lower[B]) extends AnyVal {
    def >=>[C](chain: Chain[B, C]): Chain[A, C] =
      new >=>(f, chain)
  }
}

sealed trait LowerInstances {
  implicit object monadLowerInstanceForLower extends MonadLower[Lower] with StackSafeMonad[Lower] {
      override def applicative: Applicative[Lower] =
        this

      override def monad: Monad[Lower] =
        this

      override def ask[E2 >: ExecuteContext]: Lower[E2] =
        Lower.Ask(implicitly)

      override def local[A](fa: Lower[A])(f: ExecuteContext => ExecuteContext): Lower[A] =
        Lower((ctx, s) => fa.run(f(ctx), s))

      override def get: Lower[LoweringState] =
        Lower.Get

      override def set(s: LoweringState): Lower[Unit] =
        Lower.Set(s)

      override def liftLower[A](lower: Lower[A]): Lower[A] =
        lower

      override def flatMap[A, B](fa: Lower[A])(f: A => Lower[B]): Lower[B] =
        new Lower.FlatMap[B] {
          override type Init = A
          override val init: Lower[Init] = fa
          override val bind: Init => Lower[B] = f
        }

      override def pure[A](a: A): Lower[A] =
        Lower.Pure(a)

      override def raiseError[A](e: Throwable): Lower[A] =
        Lower.Fail(e)

      override def handleErrorWith[A](fa: Lower[A])(f: Throwable => Lower[A]): Lower[A] =
        Lower.Catch(fa, f)
    }

}

