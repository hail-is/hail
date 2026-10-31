package is.hail.utils

import scala.annotation.tailrec

import cats.Monad

object StackSafe {

  def done[A](result: A): StackFrame[A] = Done(result)

  def call[A](body: => StackFrame[A]): StackFrame[A] =
    Thunk(() => body)

  @tailrec def run[A](frame: StackFrame[A]): A = frame match {
    case Done(result) => result
    case Thunk(thunk) => run(thunk())
    case more: More[_, A] => run(more.advance())
  }

  abstract class StackFrame[A] {
    def flatMap[B](f: A => StackFrame[B]): StackFrame[B] = this match {
      case Done(result) => f(result)
      case thunk: Thunk[A] =>
        val cell = new ContCell[A, B](f)
        new More[A, B](thunk, cell, cell)
      case more: More[_, A] =>
        val cell = new ContCell[A, B](f)
        more.append(cell, cell)
    }

    def map[B](f: A => B): StackFrame[B] = flatMap(a => Done(f(a)))

    final def run(): A = StackSafe.run(this)
  }

  implicit object stackFrameMonad extends Monad[StackFrame] {
    final override def pure[A](a: A): StackFrame[A] =
      done(a)

    final override def flatMap[A, B](fa: StackFrame[A])(f: A => StackFrame[B]): StackFrame[B] =
      fa.flatMap(f)

    final override def tailRecM[A, B](a: A)(f: A => StackFrame[Either[A, B]]): StackFrame[B] =
      call(f(a)).flatMap {
        case Left(a) => tailRecM(a)(f)
        case Right(b) => done(b)
      }
  }

  private class ContCell[A, B](val f: A => StackFrame[B], var next: ContCell[B, _] = null)

  final private case class Done[A](result: A) extends StackFrame[A]

  final private case class Thunk[A](force: () => StackFrame[A]) extends StackFrame[A]

  final private class More[A, B](
    _next: StackFrame[A],
    _contHead: ContCell[A, _],
    _contTail: ContCell[_, B],
  ) extends StackFrame[B] {
    // type erased locals allow mutating to different type parameters, then
    // casting `this` if needed
    private[this] var nextTE: StackFrame[_] = _next
    private[this] var contHeadTE: ContCell[_, _] = _contHead
    private[this] var contTailTE: ContCell[_, _] = _contTail

    @inline def next: StackFrame[A] = nextTE.asInstanceOf[StackFrame[A]]
    @inline def contHead: ContCell[A, _] = contHeadTE.asInstanceOf[ContCell[A, _]]
    @inline def contTail: ContCell[_, B] = contTailTE.asInstanceOf[ContCell[_, B]]

    @inline def advance(): StackFrame[B] = {
      if (contHead == null) next.asInstanceOf[StackFrame[B]]
      else next match {
        case Done(result) =>
          nextTE = contHead.f(result)
          contHeadTE = contHead.next
          this
        case thunk: Thunk[A] =>
          nextTE = thunk.force()
          this
        case more2: More[_, A] =>
          if (contHead != null)
            more2.append(contHead, contTail)
          else
            more2.asInstanceOf[More[_, B]]
      }
    }

    @inline def append[C](head: ContCell[B, _], tail: ContCell[_, C]): More[A, C] = {
      if (contHeadTE == null) {
        // A = B
        contHeadTE = head
      } else {
        contTail.next = head
      }
      contTailTE = tail
      this.asInstanceOf[More[A, C]]
    }
  }
}
