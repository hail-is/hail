package is.hail.utils

import cats.{Monad, StackSafeMonad}
import is.hail.utils.StackSafe.StackFrame

import scala.annotation.tailrec
import scala.collection.generic.{CanBuild, CanBuildFrom}

object StackSafe extends StackFrameInstances {

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

  private class ContCell[A, B](val f: A => StackFrame[B], var next: ContCell[B, _] = null)

  private final case class Done[A](result: A) extends StackFrame[A]

  private final case class Thunk[A](force: () => StackFrame[A]) extends StackFrame[A]

  private final class More[A, B](
    _next: StackFrame[A],
    _contHead: ContCell[A, _],
    _contTail: ContCell[_, B]
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

  implicit class RichIndexedSeq[A](val s: IndexedSeq[A]) extends AnyVal {
    def mapRecur[B, That](f: A => StackFrame[B])(implicit bf: CanBuildFrom[IndexedSeq[A], B, That]): StackFrame[That] = {
      val builder = bf(s)
      builder.sizeHint(s)
      var i = 0
      var cont: B => StackFrame[That] = null
      def loop(): StackFrame[That] = {
        if (i < s.size) {
          f(s(i)).flatMap(cont)
        } else {
          done(builder.result)
        }
      }
      cont = { b =>
        builder += b
        i += 1
        loop()
      }
      loop()
    }
  }

  implicit class RichArray[A](val a: Array[A]) extends AnyVal {
    def mapRecur[B](f: A => StackFrame[B])(implicit bf: CanBuildFrom[Array[A], B, Array[B]]): StackFrame[Array[B]] = {
      val builder = bf(a)
      builder.sizeHint(a)
      var i = 0
      var cont: B => StackFrame[Array[B]] = null
      def loop(): StackFrame[Array[B]] = {
        if (i < a.size) {
          f(a(i)).flatMap(cont)
        } else {
          done(builder.result)
        }
      }
      cont = { b =>
        builder += b
        i += 1
        loop()
      }
      loop()
    }
  }

  implicit class RichOption[A](val o: Option[A]) extends AnyVal {
    def mapRecur[B](f: A => StackFrame[B]): StackFrame[Option[B]] = {
      o match {
        case None => done(None)
        case Some(a) => call(f(a)).map(b => Some(b))
      }
    }
  }

  implicit class RichIterator[A](val i: Iterable[A]) extends AnyVal {
    def foreachRecur(f: A => StackFrame[Unit]): StackFrame[Unit] = {
      val it = i.iterator
      def loop(): StackFrame[Unit] = {
        if (it.hasNext) {
          f(it.next()).flatMap { _ => call(loop()) }
        } else {
          done(())
        }
      }
      loop()
    }
  }

  implicit class RichIteratorStackFrame[A](val i: Iterator[StackFrame[A]]) extends AnyVal {
    def collectRecur(implicit bf: CanBuild[A, Array[A]]): StackFrame[IndexedSeq[A]] = {
      val builder = bf()
      var cont: A => StackFrame[IndexedSeq[A]] = null
      def loop(): StackFrame[IndexedSeq[A]] = {
        if (i.hasNext) {
          i.next().flatMap(cont)
        } else {
          done(builder.result())
        }
      }
      cont = { a =>
        builder += a
        call(loop())
      }
      loop()
    }
  }

  def fillArray[A](n: Int)(body: => StackFrame[A])(implicit bf: CanBuild[A, Array[A]]): StackFrame[Array[A]] = {
    val builder = bf()
    builder.sizeHint(n)
    var i = 0
    var cont: A => StackFrame[Array[A]] = null
    def loop(): StackFrame[Array[A]] = {
      if (i < n) {
        body.flatMap(cont)
      } else {
        done(builder.result)
      }
    }
    cont = { a =>
      builder += a
      i += 1
      loop()
    }
    loop()
  }
}

sealed trait StackFrameInstances {

  implicit val stackSafeInstanceForMonad: Monad[StackFrame] =
    new Monad[StackFrame] with StackSafeMonad[StackFrame] {
      override def pure[A](x: A): StackFrame[A] =
        StackSafe.done(x)

      override def flatMap[A, B](fa: StackFrame[A])(f: A => StackFrame[B]): StackFrame[B] =
        fa.flatMap(f)
    }

}
