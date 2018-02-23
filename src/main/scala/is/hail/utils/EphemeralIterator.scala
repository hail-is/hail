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

  def valueOrElse(default: A): A =
    if (isValid) value else default

  def exhaust() { while (isValid) advance() }

  def toStagingIterator: StagingIterator[A]

  def staircased(ord: OrderingView[A]): StagingIterator[EphemeralIterator[A]] = {
    ord.setBottom()
    val stepIterator: EphemeralIterator[A] = EphemeralIterator(
      new StateMachine[A] {
        def curValue: A = self.value
        def isActive: Boolean = self.isValid && ord.isEquivalent(curValue)
        def advance() = { self.advance() }
      }
    )
    val sm = new StateMachine[EphemeralIterator[A]] {
      var isActive: Boolean = true
      val curValue: EphemeralIterator[A] = stepIterator
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
    that: EphemeralIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    mixedOrd: (A, B) => Int
  ): EphemeralIterator[Muple[EphemeralIterator[A], EphemeralIterator[B]]] = {
    this.staircased(leftOrd).orderedZipJoin(
      that.staircased(rightOrd),
      EphemeralIterator.empty,
      EphemeralIterator.empty,
      (l, r) => mixedOrd(l.head, r.head)
    )
  }

  def orderedZipJoin[B](
    that: EphemeralIterator[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int): EphemeralIterator[Muple[A, B]] = {
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
    EphemeralIterator(sm)
  }

  def innerJoinDistinct[B](
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
    that: EphemeralIterator[B],
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
}
