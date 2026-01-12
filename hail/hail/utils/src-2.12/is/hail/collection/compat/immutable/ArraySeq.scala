package is.hail.collection.compat.immutable

import scala.collection.compat.IterableOnce
import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import scala.reflect.ClassTag

object ArraySeq {
  // scalafix:off ForbiddenSymbol
  import scala.collection.compat.immutable.{ArraySeq => A}

  def empty[T <: AnyRef]: ArraySeq[T] = A.empty

  def newBuilder[T: ClassTag]: mutable.Builder[T, ArraySeq[T]] =
    Array.newBuilder[T].mapResult(unsafeWrapArray)

  def apply[T: ClassTag](elems: T*): ArraySeq[T] = from(elems)

  def from[T: ClassTag](it: IterableOnce[T]): ArraySeq[T] = it match {
    case a: mutable.WrappedArray[T] => unsafeWrapArray(a.array)
    case a: ArraySeq[T] => a
    case _ => unsafeWrapArray(it.toArray)
  }

  def unapplySeq[T](seq: ArraySeq[T]): Some[ArraySeq[T]] = Some(seq)

  def unsafeWrapArray[T](x: Array[T]): ArraySeq[T] = A.unsafeWrapArray(x)

  implicit def canBuildFrom[T: ClassTag]: CanBuildFrom[ArraySeq[_], T, ArraySeq[T]] =
    A.canBuildFrom

  def fill[T: ClassTag](n: Int)(elem: => T): ArraySeq[T] =
    unsafeWrapArray(Array.fill(n)(elem))

  def tabulate[T: ClassTag](n: Int)(f: Int => T): ArraySeq[T] =
    unsafeWrapArray(Array.tabulate(n)(f))
}
