package is.hail.utils.richUtils


class RichArray[T](val a: Array[T]) extends AnyVal {
  def index: Map[T, Int] = a.zipWithIndex.toMap
}
