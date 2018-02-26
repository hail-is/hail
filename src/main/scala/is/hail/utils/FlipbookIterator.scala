package is.hail.utils

import scala.collection.generic.Growable


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

object StagingIterator {
  def apply[A](sm: StateMachine[A]): StagingIterator[A] =
    new StagingIterator(sm)
}

class StagingIterator[A] private (sm: StateMachine[A]) extends FlipbookIterator[A] {
  private var isConsumed: Boolean = false

  // EphemeralIterator interface
  def isValid: Boolean = sm.isActive
  def value: A = { assert(isValid && !isConsumed); sm.curValue }
  def advance(): Unit = {
    assert(isValid)
    sm.advance()
    isConsumed = false
  }

  // Additional StagingIterator methods
  def consume(): A = { assert(isValid && !isConsumed); isConsumed = true; sm.curValue }
  def stage(): Unit = {
    if (isConsumed) {
      sm.advance()
      isConsumed = false
    }
  }
  def consumedValue: A = { assert(isValid && isConsumed); sm.curValue }

  // (Buffered)Iterator interface, not intended to be used directly, only for
  // passing a StagingIterator where an Iterator is expected
  def head: A = sm.curValue
  def hasNext: Boolean = {
    if (isValid) {
      stage()
      isValid
    } else false
  }
  def next(): A = {
    stage()
    assert(isValid)
    consume()
  }

  def toStagingIterator: StagingIterator[A] = this
}

object FlipbookIterator {
  def apply[A](sm: StateMachine[A]): FlipbookIterator[A] =
    StagingIterator(sm)

  def empty[A] = StagingIterator(StateMachine.terminal[A])
}

/**
  * The primary public interface of EphemeralIterator[A] consists of the methods
  * - isValid: Bolean
  * - value: A
  * - advance(): Unit
  *
  * It also extends BufferedIterator[A] for interoperability with Scala and
  * Spark
  *
  * To define a new FlipbookIterator, define a StateMachine (which has the same
  * abstract methods as FlipbookIterator, but is unchecked), then use the
  * factory method FlipbookIterator(sm).
  */
abstract class FlipbookIterator[A] extends BufferedIterator[A] { self =>
  def isValid: Boolean
  def value: A
  def advance(): Unit

  def valueOrElse(default: A): A =
    if (isValid) value else default

  def exhaust() { while (isValid) advance() }

  def toStagingIterator: StagingIterator[A]

  def staircased(ord: OrderingView[A]): StagingIterator[FlipbookIterator[A]] = {
    ord.setBottom()
    val stepIterator: FlipbookIterator[A] = FlipbookIterator(
      new StateMachine[A] {
        def curValue: A = self.value
        def isActive: Boolean = self.isValid && ord.isEquivalent(curValue)
        def advance() = { self.advance() }
      }
    )
    val sm = new StateMachine[FlipbookIterator[A]] {
      var isActive: Boolean = true
      val curValue: FlipbookIterator[A] = stepIterator
      def advance() = {
        stepIterator.exhaust()
        if (self.isValid) {
          ord.setValue(self.value)
        }
        else {
          ord.setBottom()
          isActive = false
        }
      }
    }
    sm.advance()
    StagingIterator(sm)
  }

  def cogroup[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    mixedOrd: (A, B) => Int
  ): FlipbookIterator[Muple[FlipbookIterator[A], FlipbookIterator[B]]] = {
    this.staircased(leftOrd).orderedZipJoin(
      that.staircased(rightOrd),
      FlipbookIterator.empty,
      FlipbookIterator.empty,
      (l, r) => mixedOrd(l.head, r.head)
    )
  }

  def orderedZipJoin[B](
    that: FlipbookIterator[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int): FlipbookIterator[Muple[A, B]] = {
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
              mixedOrd(left.value, right.value)
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
    FlipbookIterator(sm)
  }

  def innerJoinDistinct[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    for { Muple(l, r) <- this.cogroup(that, leftOrd, rightOrd, mixedOrd) if r.isValid
          lrv <- l
    } yield result.set(lrv, r.value)
  }

  def leftJoinDistinct[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    for { Muple(l, r) <- this.cogroup(that, leftOrd, rightOrd, mixedOrd)
          lrv <- l
    } yield result.set(lrv, r.valueOrElse(rightDefault))
  }

  def innerJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      lIt.product(rIt, rightBuffer, result)
    }
  }

  def leftJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (rIt.isValid) lIt.product(rIt, rightBuffer, result)
      else lIt.map( lElem => result.set(lElem, rightDefault) )
    }
  }

  def rightJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (lIt.isValid) lIt.product(rIt, rightBuffer, result)
      else rIt.map( rElem => result.set(leftDefault, rElem) )
    }
  }

  def outerJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int
  ): Iterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (!lIt.isValid) rIt.map( rElem => result.set(leftDefault, rElem) )
      else if (!rIt.isValid) lIt.map( lElem => result.set(lElem, rightDefault) )
      else lIt.product(rIt, rightBuffer, result)
    }
  }

  private def product[B](
    that: FlipbookIterator[B],
    buffer: Growable[B] with Iterable[B],
    result: Muple[A, B]
  ): Iterator[Muple[A, B]] = {
    buffer.clear()
    if (this.isValid) buffer ++= that //avoid copying right iterator when not needed
    for { lElem <- this
          rElem <- buffer
    } yield result.set(lElem, rElem)
  }

  def compareUsing[B](that: Iterator[B], eq: (A, B) => Boolean): Boolean = {
    while (this.hasNext && that.hasNext)
      if (!eq(this.next(), that.next()))
        return false
    !this.hasNext && !that.hasNext
  }

  // head, next, and hasNext are not meant to be used directly, only to enable
  // EphemeralIterator to be used where an Iterator is expected.
  def head: A
  def next(): A
  def hasNext: Boolean
}
