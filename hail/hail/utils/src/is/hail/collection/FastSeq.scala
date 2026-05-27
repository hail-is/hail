package is.hail.collection

import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterable

import scala.reflect.ClassTag

object FastSeq {
  @inline def empty[T]: IndexedSeq[T] =
    ArraySeq.empty

  def apply[T: ClassTag](args: T*): IndexedSeq[T] =
    if (args.isEmpty) empty else args.toFastSeq
}
