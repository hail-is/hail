package is.hail.utils

import scala.collection.mutable
import scala.reflect.ClassTag

object FastSeq {
  def empty[T](implicit tct: ClassTag[T]): Seq[T] = FastSeq()

  def apply[T](args: T*)(implicit tct: ClassTag[T]): Seq[T] = {
    assert(args.isInstanceOf[mutable.WrappedArray[_]])
    args
  }
}
