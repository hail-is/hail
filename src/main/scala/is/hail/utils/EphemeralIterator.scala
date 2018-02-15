package is.hail.utils

/**
  * The primary public interface of EphemeralIterator[A] consists of the methods
  * - isValid: Bolean
  * - value: A
  * - advance(): Unit
  *
  * It also extends BufferedIterator[A] for interoperability with Scala and
  * Spark
  */
abstract class EphemeralIterator[A] extends BufferedIterator[A] { self =>
  // There are three abstract methods that must be implemented to define an
  // EphemeralIterator
  def isValid: Boolean
  def head: A
  protected def advanceHead(): Unit

  def value: A = { assert(isValid && !_isConsumed); head }
  def advance(): Unit = {
    assert(isValid)
    advanceHead()
    _isConsumed = false
  }

  private var _isConsumed: Boolean = false
  protected def isConsumed: Boolean = _isConsumed
  protected def consume(): A = { assert(isValid && !_isConsumed); _isConsumed = true; head }
  protected def stage(): Unit = {
    if (_isConsumed) {
      advanceHead()
      _isConsumed = false
    }
  }

  def next(): A = {
    stage()
    assert(isValid)
    consume()
  }

  def hasNext: Boolean = {
    if (isValid) {
      stage()
      isValid
    } else false
  }

  def toStagingIterator: StagingIterator[A] = new StagingIterator[A] {
    def isValid = self.isValid
    def head = self.head
    def advanceHead() = self.advanceHead()
  }

  def staircased(equiv: EquivalenceClassView[A]): StagingIterator[EphemeralIterator[A]] =
    new StaircaseIterator(self, equiv)

  def compareUsing[B](that: Iterator[B], eq: (A, B) => Boolean): Boolean = {
    while (this.hasNext && that.hasNext)
      if (!eq(this.next(), that.next()))
        return false
    !this.hasNext && !that.hasNext
  }
}

object EphemeralIterator {
  def empty[A] = new EphemeralIterator[A] {
    def isValid = false
    def head = throw new NoSuchElementException("head on empty iterator")
    def advanceHead() {}
  }
}

abstract class StagingIterator[A] extends EphemeralIterator[A] {
  override def isConsumed: Boolean = super.isConsumed
  override def consume(): A = super.consume()
  def consumedValue: A = { assert(isValid && isConsumed); head }
  override def stage(): Unit = super.stage()
}
