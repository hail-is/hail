package is.hail.collection

import is.hail.collection.Indexed

import scala.collection.immutable.ArraySeq
import scala.collection.mutable

import cats.{Applicative, Eval, Foldable, Monad, Traverse}
import cats.data.Chain

package object implicits {
  implicit object indexedSeqTraverse extends Traverse[IndexedSeq] {
    final override def traverse[G[_], A, B](
      fa: IndexedSeq[A]
    )(
      f: A => G[B]
    )(implicit
      G: Applicative[G]
    ): G[IndexedSeq[B]] =
      G.map(Chain.traverseViaChain(fa)(f))(_.iterator.to(ArraySeq.untagged))

    final override def map[A, B](fa: IndexedSeq[A])(f: A => B): IndexedSeq[B] =
      fa.map(f)

    final override def foldLeft[A, B](fa: IndexedSeq[A], b: B)(f: (B, A) => B): B =
      fa.foldLeft(b)(f)

    final override def foldRight[A, B](fa: IndexedSeq[A], lb: Eval[B])(f: (A, Eval[B]) => Eval[B])
      : Eval[B] =
      Foldable.iterateRight(fa, lb)(f)

    final override def foldM[G[_], A, B](
      fa: IndexedSeq[A],
      z: B,
    )(
      f: (B, A) => G[B]
    )(implicit
      G: Monad[G]
    ): G[B] = {
      val L = fa.length
      if (L == 0) G.pure(z)
      else G.tailRecM(new Indexed(0, z)) { t =>
        val i = t.i; val b0 = t.value;
        if (i == L - 1) G.map(f(b0, fa(i)))(Right(_))
        else G.map(f(b0, fa(i)))(b1 => Left(new Indexed(i + 1, b1)))
      }
    }
  }

  // Range isn't an F[A], so cats' syntax can't find indexedSeqTraverse for it
  implicit def rangeTraverseOps(r: Range): Traverse.AllOps[IndexedSeq, Int] =
    new Traverse.AllOps[IndexedSeq, Int] {
      type TypeClassType = Traverse[IndexedSeq]
      val self: IndexedSeq[Int] = r
      val typeClassInstance: Traverse[IndexedSeq] = indexedSeqTraverse
    }

  implicit def toRichArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichArraySeqCompanion(a: ArraySeq.type): RichArraySeqCompanion.type =
    RichArraySeqCompanion

  implicit def toRichIndexedSeq[T](s: IndexedSeq[T]): RichIndexedSeq[T] = new RichIndexedSeq(s)

  implicit def toRichIndexedSeqAnyRef[T <: AnyRef](s: IndexedSeq[T]): RichIndexedSeqAnyRef[T] =
    new RichIndexedSeqAnyRef(s)

  implicit def arrayToRichIndexedSeq[T](s: Array[T]): RichIndexedSeq[T] =
    new RichIndexedSeq(ArraySeq.unsafeWrapArray(s))

  implicit def toRichIterable[T](i: Iterable[T]): RichIterable[T] = new RichIterable(i)

  implicit def toRichIterable[T](a: Array[T]): RichIterable[T] = new RichIterable(a)

  implicit def toRichIterator[T](it: Iterator[T]): RichIterator[T] = new RichIterator[T](it)

  implicit def toRichMap[K, V](m: Map[K, V]): RichMap[K, V] = new RichMap(m)

  implicit def toRichMutableMap[K, V](m: mutable.Map[K, V]): RichMutableMap[K, V] =
    new RichMutableMap(m)

  implicit def toRichOption[T](o: Option[T]): RichOption[T] = new RichOption[T](o)

  implicit def toRichOrderedArray[T: Ordering](a: Array[T]): RichOrderedArray[T] =
    new RichOrderedArray(a)

  implicit def toRichOrderedSeq[T: Ordering](s: Seq[T]): RichOrderedSeq[T] =
    new RichOrderedSeq[T](s)

}
