package is.hail.utils

import is.hail.collection.implicits.toRichIterable

import java.io.InputStream

package object implicits {

  implicit def toRichBoolean(b: Boolean): RichBoolean = new RichBoolean(b)

  implicit def toRichContextIterator[T](it: Iterator[WithContext[T]]): RichContextIterator[T] =
    new RichContextIterator[T](it)

  implicit def toRichPredicate[A](f: A => Boolean): RichPredicate[A] = new RichPredicate[A](f)

  implicit def toRichString(str: String): RichString = new RichString(str)

  implicit def toTruncatable(s: String): Truncatable = s.truncatable()

  implicit def toTruncatable[T](it: Iterable[T]): Truncatable = it.truncatable()

  implicit def toTruncatable(arr: Array[_]): Truncatable = toTruncatable(arr: Iterable[_])

  implicit def toRichInputStream(in: InputStream): RichInputStream = new RichInputStream(in)

  implicit def toRichPartialKleisliOptionFunction[A, B](x: PartialFunction[A, Option[B]])
    : RichPartialKleisliOptionFunction[A, B] = new RichPartialKleisliOptionFunction(x)

}
