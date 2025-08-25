package is.hail.utils.compat.immutable

import scala.collection.generic.CanBuildFrom
import scala.collection.mutable
import scala.reflect.ClassTag

object ArraySeq {
  private val A = scala.collection.compat.immutable.ArraySeq

  private val EmptyArraySeq = new A.ofRef[AnyRef](new Array[AnyRef](0))
  def empty[T <: AnyRef]: ArraySeq[T] = EmptyArraySeq.asInstanceOf[ArraySeq[T]]

  def newBuilder[T](implicit elemTag: ClassTag[T]): mutable.Builder[T, ArraySeq[T]] =
    Array.newBuilder[T].mapResult(unsafeWrapArray)

  def apply[T](elems: T*)(implicit elemTag: ClassTag[T]): ArraySeq[T] = elems match {
    case a: mutable.WrappedArray[T] => unsafeWrapArray(a.array)
    case _ =>
      val b = newBuilder[T]
      b ++= elems
      b.result()
  }

  def unapplySeq[T](seq: ArraySeq[T]): Some[ArraySeq[T]] = Some(seq)

  def unsafeWrapArray[T](x: Array[T]): ArraySeq[T] = A.unsafeWrapArray(x)

  implicit def canBuildFrom[T: ClassTag]: CanBuildFrom[ArraySeq[_], T, ArraySeq[T]] =
    new CanBuildFrom[ArraySeq[_], T, ArraySeq[T]] {
      def apply(from: ArraySeq[_]): mutable.Builder[T, ArraySeq[T]] =
        Array.newBuilder[T].mapResult(unsafeWrapArray)
      def apply: mutable.Builder[T, ArraySeq[T]] =
        Array.newBuilder[T].mapResult(unsafeWrapArray)
    }
}
