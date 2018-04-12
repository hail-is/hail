package is.hail.utils

class SetupIterator[T](it: Iterator[T], setup: () => Unit) extends Iterator[T] {
  private[this] var needsSetup = true

  def hasNext: Boolean = {
    if (needsSetup) {
      setup()
      needsSetup = false
    }
    it.hasNext
  }

  def next(): T = {
    if (needsSetup)
      setup()
    needsSetup = true
    it.next()
  }
}
