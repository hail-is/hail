package is.hail.utils

trait StateMachine[A] {
  def isActive: Boolean
  def curValue: A
  def advance(): Unit
}

object StateMachine {
  def terminal[A]: StateMachine[A] = new StateMachine[A] {
    val isActive = false
    var curValue: A = _
    def advance() {}
  }
}

object EphemeralIterator {
  def apply[A](sm: StateMachine[A]): EphemeralIterator[A] =
    StagingIterator(sm).toEphemeralIterator

  def empty[A] = StagingIterator(StateMachine.terminal[A])
}

object StagingIterator {
  def apply[A](sm: StateMachine[A]): StagingIterator[A] =
    new StagingIterator(sm)
}

class StagingIterator[A] private (sm: StateMachine[A]) extends EphemeralIterator[A] {
  def head: A = sm.curValue
  def isValid: Boolean = sm.isActive

  private var isConsumed: Boolean = false
  def consume(): A = { assert(isValid && !isConsumed); isConsumed = true; head }
  def stage(): Unit = {
    if (isConsumed) {
      sm.advance()
      isConsumed = false
    }
  }
  def consumedValue: A = { assert(isValid && isConsumed); head }

  def value: A = { assert(isValid && !isConsumed); head }
  def advance(): Unit = {
    assert(isValid)
    sm.advance()
    isConsumed = false
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

  def toStagingIterator: StagingIterator[A] = this
}

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
  def value: A
  def advance(): Unit

  def head: A
  def next(): A
  def hasNext: Boolean

  def exhaust() { while (isValid) advance() }

  def toStagingIterator: StagingIterator[A]

  def staircased(equiv: EquivalenceClassView[A]): StagingIterator[EphemeralIterator[A]] = {
    equiv.setEmpty()
    val stepIterator: EphemeralIterator[A] = EphemeralIterator(
      new StateMachine[A] {
        def curValue: A = self.value
        def isActive: Boolean = self.isValid && equiv.inEquivClass(curValue)
        def advance() = { self.advance() }
      }
    )
    val sm = new StateMachine[EphemeralIterator[A]] {
      var isActive: Boolean = true
      val curValue: EphemeralIterator[A] = stepIterator
      def advance() = {
        stepIterator.exhaust()
        if (self.isValid) {
          equiv.setEquivClass(self.value)
        }
        else {
          equiv.setEmpty()
          isActive = false
        }
      }
    }
    sm.advance()
    StagingIterator(sm)
  }

  def orderedZipJoin[B](
    that: EphemeralIterator[B],
    leftDefault: A,
    rightDefault: B,
    ordering: (A, B) => Int): EphemeralIterator[Muple[A, B]] = {
    val left = self.toStagingIterator
    val right = that.toStagingIterator
    val sm = new StateMachine[Muple[A, B]] {
      val curValue = Muple(leftDefault, rightDefault)
      var isActive = true
      def advance() {
        left.stage()
        right.stage()
        val c = {
          if (left.isValid) {
            if (right.isValid)
              ordering(left.value, right.value)
            else
              -1
          } else if (right.isValid)
              1
            else {
              isActive = false
              return
            }
        }
        if (c == 0)
          curValue.set(left.consume(), right.consume())
        else if (c < 0)
          curValue.set(left.consume(), rightDefault)
        else
          // c > 0
          curValue.set(leftDefault, right.consume())
      }
    }

    sm.advance()
    EphemeralIterator(sm)
  }

  def compareUsing[B](that: Iterator[B], eq: (A, B) => Boolean): Boolean = {
    while (this.hasNext && that.hasNext)
      if (!eq(this.next(), that.next()))
        return false
    !this.hasNext && !that.hasNext
  }
}
