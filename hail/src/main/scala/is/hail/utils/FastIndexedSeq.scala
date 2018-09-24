package is.hail.utils

import scala.collection.mutable
import scala.reflect.ClassTag

object FastIndexedSeq {
  def empty[T](implicit tct: ClassTag[T]): IndexedSeq[T] = FastIndexedSeq()

  def apply[T](args: T*)(implicit tct: ClassTag[T]): IndexedSeq[T] = {
    args match {
      case args: mutable.WrappedArray[T] => args
      case args: mutable.ArrayBuffer[T] => args
      case _ => args.toArray[T]
    }
  }
}
