package is.hail.utils

import scala.collection.mutable
import scala.reflect.ClassTag

object FastIndexedSeq {
  def empty[T](implicit tct: ClassTag[T]): IndexedSeq[T] = FastIndexedSeq()

  def apply[T](args: T*)(implicit tct: ClassTag[T]): IndexedSeq[T] =
    args.asInstanceOf[mutable.WrappedArray[T]]
}
