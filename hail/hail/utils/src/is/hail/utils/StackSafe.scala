package is.hail.utils

import scala.annotation.tailrec

import cats.StackSafeMonad

object StackSafe {

  def done[A](result: A): StackFrame[A] = Done(result)

  def call[A](body: => StackFrame[A]): StackFrame[A] =
    Thunk(() => body)

  /** Runs `frame` in constant JVM stack. Pending continuations (`FlatMapped` and `Mapped` nodes)
    * live on a `List` stack, innermost at the head. A `Mapped` result stays a raw value, so no
    * `Done` is allocated between `map`s.
    */
  def run[A](frame: StackFrame[A]): A = {
    @tailrec def go(cur: StackFrame[_], stack: List[StackFrame[_]]): Any = cur match {
      case fm: FlatMapped[_, _] => go(fm.source, fm :: stack)
      case m: Mapped[_, _] => go(m.source, m :: stack)
      case Done(v) =>
        var value: Any = v
        var s = stack
        while (s.nonEmpty && s.head.isInstanceOf[Mapped[_, _]]) {
          value = s.head.asInstanceOf[Mapped[Any, Any]].f(value)
          s = s.tail
        }
        if (s.isEmpty) value
        else go(s.head.asInstanceOf[FlatMapped[Any, Any]].f(value), s.tail)
      case Thunk(force) => go(force(), stack)
    }
    go(frame, Nil).asInstanceOf[A]
  }

  abstract class StackFrame[A] {
    def flatMap[B](f: A => StackFrame[B]): StackFrame[B] =
      this match {
        case Done(result) => f(result)
        case _ => new FlatMapped(this, f)
      }

    def map[B](f: A => B): StackFrame[B] =
      this match {
        case Done(result) => Done(f(result))
        case _ => new Mapped(this, f)
      }

    final def run(): A = StackSafe.run(this)
  }

  implicit object stackFrameMonad extends StackSafeMonad[StackFrame] {
    final override def pure[A](a: A): StackFrame[A] =
      done(a)

    // Unlike `StackFrame.flatMap`, this never runs `f` at once, even on `Done`, so `flatMap`
    // through this instance is stack-safe by definition, as `StackSafeMonad` requires.
    final override def flatMap[A, B](fa: StackFrame[A])(f: A => StackFrame[B]): StackFrame[B] =
      new FlatMapped(fa, f)

    // `f` returns a value, not a frame, so running it at once can't recurse.
    final override def map[A, B](fa: StackFrame[A])(f: A => B): StackFrame[B] =
      fa.map(f)

    // `StackSafeMonad`'s default runs the first `f(a)` at once and allocates a new continuation
    // every step. This suspends the first step and shares one continuation across all steps.
    final override def tailRecM[A, B](a: A)(f: A => StackFrame[Either[A, B]]): StackFrame[B] = {
      def k(e: Either[A, B]): StackFrame[B] =
        e match {
          case Left(a) => new FlatMapped(f(a), k)
          case Right(b) => done(b)
        }

      new FlatMapped(Done[Either[A, B]](Left(a)), k)
    }
  }

  final private case class Done[A](result: A) extends StackFrame[A]

  final private case class Thunk[A](force: () => StackFrame[A]) extends StackFrame[A]

  final private class FlatMapped[X, A](val source: StackFrame[X], val f: X => StackFrame[A])
      extends StackFrame[A]

  final private class Mapped[X, A](val source: StackFrame[X], val f: X => A) extends StackFrame[A]
}
