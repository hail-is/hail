package is.hail.utils

class SetupIterator[T](it: Iterator[T], setup: () => Unit) extends Iterator[T] {
  private[this] var needsSetup = true

  private[this] def maybeSetup(): Unit = {
    if (needsSetup) {
      setup()
      needsSetup = false
    }
  }

  def hasNext: Boolean = {
    maybeSetup()
    it.hasNext
  }

  def next(): T = {
    maybeSetup()
    it.next()
  }
}
