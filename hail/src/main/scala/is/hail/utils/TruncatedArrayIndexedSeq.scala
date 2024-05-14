package is.hail.utils

class TruncatedArrayIndexedSeq[T](a: Array[T], newLength: Int)
    extends IndexedSeq[T] with Serializable {
  def length: Int = newLength

  def apply(idx: Int): T = {
    if (idx < 0 || idx >= newLength)
      throw new IndexOutOfBoundsException(idx.toString)
    a(idx)
  }
}
