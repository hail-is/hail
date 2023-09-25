package is.hail.utils

import scala.reflect.ClassTag

object FastSeq {
  def empty[T: ClassTag]: IndexedSeq[T] =
    FastSeq()

  def apply[T: ClassTag](args: T*): IndexedSeq[T] =
    args.toFastSeq
}
