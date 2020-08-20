package is.hail.utils

import scala.annotation.tailrec
import scala.collection.generic.{CanBuild, CanBuildFrom}

object StackSafe {

  def done[A](result: A): StackFrame[A] = Done(result)

  def call[A](body: => StackFrame[A]): StackFrame[A] =
    Thunk(() => body)

  object StackFrame {
    @tailrec def run[A](frame: StackFrame[A]): A = frame match {
      case Done(result) => result
      case Thunk(thunk) => run(thunk())
      case more: More[x, A] => run(more.advance())
    }
  }

  abstract class StackFrame[A] {
    def flatMap[B](f: A => StackFrame[B]): StackFrame[B] = this match {
      case Done(result) => f(result)
      case thunk: Thunk[A] =>
        val cell = new ContCell[A, B](f)
        new More[A, B] (thunk, cell, cell)
      case more: More[x, A] =>
        val cell = new ContCell[A, B](f)
        more.append(cell, cell)
    }

    def map[B](f: A => B): StackFrame[B] = flatMap(a => Done(f(a)))

    final def run(): A = StackFrame.run(this)
  }

  private class ContCell[A, B](val f: A => StackFrame[B], var next: ContCell[B, _] = null)

  private final case class Done[A](result: A) extends StackFrame[A]

  private final case class Thunk[A](force: () => StackFrame[A]) extends StackFrame[A]

  private final class More[A, B](
    // private vars don't create getters/setters
    private var next: StackFrame[A],
    private var contHead: ContCell[A, _],
    private var contTail: ContCell[_, B]
  ) extends StackFrame[B] {

    @inline def advance(): StackFrame[B] = {
      if (contHead == null) next.asInstanceOf[StackFrame[B]]
      else next match {
        case Done(result) =>
          contHead match {
            case head: ContCell[A, x] =>
              val ret = this.asInstanceOf[More[x, B]]
              ret.next = head.f(result)
              ret.contHead = head.next
              ret
          }
        case thunk: Thunk[A] =>
          next = thunk.force()
          this
        case more2: More[_, A] =>
          if (contHead != null)
            more2.append(contHead, contTail)
          else
            more2.asInstanceOf[More[_, B]]
      }
    }

    @inline def append[C](head: ContCell[B, _], tail: ContCell[_, C]): More[A, C] = {
      val resultM = this.asInstanceOf[More[A, C]]
      if (contHead == null) {
        // A = B
        resultM.contHead = head.asInstanceOf[ContCell[A, _]]
      } else {
        contTail.next = head
      }
      resultM.contTail = tail
      resultM
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

  implicit class RichIterator[A](val i: Iterator[StackFrame[A]]) extends AnyVal {
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
