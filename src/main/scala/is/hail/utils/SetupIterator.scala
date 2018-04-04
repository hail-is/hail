package is.hail.utils

class SetupIterator[T](it: Iterator[T], setup: () => Unit) extends Iterator[T] {
  private[this] var needsSetup = false

  private[this] def maybeSetup(): Unit = {
    if (needsSetup) {
      setup()
      needsSetup = true
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
