package is.hail.utils

import scala.reflect.ClassTag

object FastSeq {
  def empty[T: ClassTag]: IndexedSeq[T] =
    IndexedSeq.empty

  def apply[T: ClassTag](args: T*): IndexedSeq[T] =
    if (args.isEmpty) empty else args.toFastSeq
}
