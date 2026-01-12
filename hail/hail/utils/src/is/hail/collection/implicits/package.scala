package is.hail.collection

import is.hail.collection.compat.immutable.ArraySeq

import scala.collection.mutable

package object implicits {
  implicit def toRichArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichArraySeqCompanion(a: ArraySeq.type): RichArraySeqCompanion.type =
    RichArraySeqCompanion

  implicit def toRichIndexedSeq[T](s: IndexedSeq[T]): RichIndexedSeq[T] = new RichIndexedSeq(s)

  implicit def toRichIndexedSeqAnyRef[T <: AnyRef](s: IndexedSeq[T]): RichIndexedSeqAnyRef[T] =
    new RichIndexedSeqAnyRef(s)

  implicit def arrayToRichIndexedSeq[T](s: Array[T]): RichIndexedSeq[T] = new RichIndexedSeq(s)

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
