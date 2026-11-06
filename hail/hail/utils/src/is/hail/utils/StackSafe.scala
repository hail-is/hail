package is.hail.utils

import scala.annotation.tailrec
import scala.collection.{BuildFrom, Factory}
import scala.collection.immutable.ArraySeq

object StackSafe {

  def done[A](result: A): StackFrame[A] = new Done(result)
  def call[A](body: => StackFrame[A]): StackFrame[A] = new Thunk(() => body)

  def run[A](frame: StackFrame[A]): A = {
    @tailrec def go(cur: StackFrame[_], stack: List[StackFrame[_]]): Any =
      cur match {
        case fm: FlatMapped[_, _] => go(fm.source, fm :: stack)
        case m: Mapped[_, _] => go(m.source, m :: stack)
        case d: Done[_] =>
          var value: Any = d.result
          var s = stack
          while (s.nonEmpty && s.head.isInstanceOf[Mapped[_, _]]) {
            value = s.head.asInstanceOf[Mapped[Any, Any]].f(value)
            s = s.tail
          }
          if (s.isEmpty) value
          else go(s.head.asInstanceOf[FlatMapped[Any, Any]].f(value), s.tail)
        case t: Thunk[_] => go(t.force(), stack)
      }

    go(frame, Nil).asInstanceOf[A]
  }

  sealed abstract class StackFrame[A] {
    final def flatMap[B](f: A => StackFrame[B]): StackFrame[B] =
      this match {
        case d: Done[A] => f(d.result)
        case _ => new FlatMapped(this, f)
      }

    final def map[B](f: A => B): StackFrame[B] =
      this match {
        case d: Done[A] => done(f(d.result))
        case _ => new Mapped(this, f)
      }

    final def run(): A = StackSafe.run(this)
  }

  final private class Done[A](val result: A) extends StackFrame[A]

  final private class Thunk[A](val force: () => StackFrame[A]) extends StackFrame[A]

  final private class FlatMapped[X, A](val source: StackFrame[X], val f: X => StackFrame[A])
      extends StackFrame[A]

  final private class Mapped[X, A](val source: StackFrame[X], val f: X => A) extends StackFrame[A]

  implicit class RichIndexedSeq[A](private val s: IndexedSeq[A]) extends AnyVal {
    def mapRecur[B, That](f: A => StackFrame[B])(implicit bf: BuildFrom[IndexedSeq[A], B, That])
      : StackFrame[That] = {
      val builder = bf.newBuilder(s)
      builder.sizeHint(s)
      var i = 0
      var step: StackFrame[That] = null

      def loop(): StackFrame[That] =
        if (i < s.size) step
        else done(builder.result())

      val k: B => StackFrame[That] = { b =>
        builder += b
        i += 1
        loop()
      }
      step = call(f(s(i)).flatMap(k))
      loop()
    }
  }

  implicit class RichArray[A](private val a: Array[A]) extends AnyVal {
    def mapRecur[B](f: A => StackFrame[B])(implicit bf: BuildFrom[Array[A], B, Array[B]])
      : StackFrame[Array[B]] = {
      val builder = bf.newBuilder(a)
      builder.sizeHint(a)
      var i = 0
      var step: StackFrame[Array[B]] = null

      def loop(): StackFrame[Array[B]] =
        if (i < a.length) step
        else done(builder.result())

      val k: B => StackFrame[Array[B]] = { b =>
        builder += b
        i += 1
        loop()
      }

      step = call(f(a(i)).flatMap(k))
      loop()
    }
  }

  implicit class RichOption[A](private val o: Option[A]) extends AnyVal {
    def mapRecur[B](f: A => StackFrame[B]): StackFrame[Option[B]] =
      o match {
        case None => done(None)
        case Some(a) => call(f(a)).map(b => Some(b))
      }
  }

  implicit class RichIterator[A](private val i: Iterable[A]) extends AnyVal {
    def foreachRecur(f: A => StackFrame[Unit]): StackFrame[Unit] = {
      val it = i.iterator
      var step: StackFrame[Unit] = null

      def loop(): StackFrame[Unit] =
        if (it.hasNext) step else done(())

      val k: Unit => StackFrame[Unit] = _ => loop()

      step = call(f(it.next()).flatMap(k))
      loop()
    }
  }

  implicit class RichIteratorStackFrame[A](private val i: Iterator[StackFrame[A]]) extends AnyVal {
    def collectRecur(implicit bf: Factory[A, ArraySeq[A]]): StackFrame[IndexedSeq[A]] = {
      val builder = bf.newBuilder
      var step: StackFrame[IndexedSeq[A]] = null

      def loop(): StackFrame[IndexedSeq[A]] =
        if (i.hasNext) step else done(builder.result())

      val k: A => StackFrame[IndexedSeq[A]] = { a =>
        builder += a
        loop()
      }

      step = call(i.next().flatMap(k))
      loop()
    }
  }

  def fillArray[A, C](n: Int)(body: => StackFrame[A])(implicit bf: Factory[A, C]): StackFrame[C] = {
    val builder = bf.newBuilder
    builder.sizeHint(n)
    var i = 0
    var step: StackFrame[C] = null

    def loop(): StackFrame[C] =
      if (i < n) step else done(builder.result())

    val k: A => StackFrame[C] = { a =>
      builder += a
      i += 1
      loop()
    }

    step = call(body.flatMap(k))
    loop()
  }
}
