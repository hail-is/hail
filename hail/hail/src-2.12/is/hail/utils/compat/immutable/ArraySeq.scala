package is.hail.utils.compat.immutable

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import scala.reflect.ClassTag

object ArraySeq {
  import scala.collection.compat.immutable.{ArraySeq => A}

  def empty[T <: AnyRef]: ArraySeq[T] = A.empty

  def newBuilder[T: ClassTag]: mutable.Builder[T, ArraySeq[T]] =
    Array.newBuilder[T].mapResult(unsafeWrapArray)

  def apply[T: ClassTag](elems: T*): ArraySeq[T] = elems match {
    case a: mutable.WrappedArray[T] => unsafeWrapArray(a.array)
    case _ => unsafeWrapArray(elems.toArray)
  }

  def unapplySeq[T](seq: ArraySeq[T]): Some[ArraySeq[T]] = Some(seq)

  def unsafeWrapArray[T](x: Array[T]): ArraySeq[T] = A.unsafeWrapArray(x)

  implicit def canBuildFrom[T: ClassTag]: CanBuildFrom[ArraySeq[_], T, ArraySeq[T]] =
    A.canBuildFrom
}
