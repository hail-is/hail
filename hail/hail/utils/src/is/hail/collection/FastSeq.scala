package is.hail.collection

import is.hail.collection.implicits.toRichIterable

import scala.reflect.ClassTag

object FastSeq {
  def empty[T]: IndexedSeq[T] =
    IndexedSeq.empty

  def apply[T: ClassTag](args: T*): IndexedSeq[T] =
    if (args.isEmpty) empty else args.toFastSeq
}
