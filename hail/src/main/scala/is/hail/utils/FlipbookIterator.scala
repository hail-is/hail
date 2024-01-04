package is.hail.utils

import scala.collection.GenTraversableOnce
import scala.collection.generic.Growable
import scala.collection.mutable.PriorityQueue
import scala.reflect.ClassTag

/** A StateMachine has the same primary interface as FlipbookIterator, but the implementations are
  * not expected to be checked (for instance, value does not need to assert isValid). The only
  * intended use of a StateMachine is to instantiate a FlipbookIterator or StagingIterator through
  * the corresponding factory methods.
  *
  * A StateMachine implementation must satisfy the following properties:
  *   - isValid and value do not change the state of the StateMachine in any observable way. In
  *     other words, if advance() is not called, then any number of calls to value and isValid will
  *     always have the same return values.
  *   - If isValid is true, than value returns a valid value. If isValid is false, then the behavior
  *     of value is undefined.
  *   - advance() puts the StateMachine into a new state, after which the return values of isValid
  *     and value may have changed.
  */
abstract class StateMachine[A] {
  def isValid: Boolean
  def value: A
  def advance(): Unit
}

object StateMachine {
  def terminal[A]: StateMachine[A] =
    new StateMachine[A] {
      override val isValid = false
      override def value: A = throw new NoSuchElementException()
      override def advance(): Unit = {}
    }
}

object StagingIterator {
  def apply[A](sm: StateMachine[A]): StagingIterator[A] =
    new StagingIterator(sm)
}

class StagingIterator[A] private (sm: StateMachine[A]) extends FlipbookIterator[A] {
  private var isConsumed: Boolean = false

  // FlipbookIterator interface
  def isValid: Boolean = sm.isValid
  def value: A = { assert(isValid && !isConsumed); sm.value }

  def advance(): Unit = {
    assert(isValid)
    isConsumed = false
    sm.advance()
  }

  // Additional StagingIterator methods
  def consume(): A = { assert(isValid && !isConsumed); isConsumed = true; sm.value }

  def stage(): Unit =
    if (isConsumed) {
      isConsumed = false
      sm.advance()
    }

  def consumedValue: A = { assert(isValid && isConsumed); sm.value }

  // (Buffered)Iterator interface, not intended to be used directly, only for
  // passing a StagingIterator where an Iterator is expected
  def head: A = { stage(); sm.value }

  def hasNext: Boolean =
    if (isValid) {
      stage()
      isValid
    } else false

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

  def multiZipJoin[A: ClassTag](
    its: Array[FlipbookIterator[A]],
    ord: (A, A) => Int,
  ): FlipbookIterator[BoxedArrayBuilder[(A, Int)]] = {
    object TmpOrd extends Ordering[(A, Int)] {
      def compare(x: (A, Int), y: (A, Int)): Int = ord(y._1, x._1)
    }
    val sm = new StateMachine[BoxedArrayBuilder[(A, Int)]] {
      val q: PriorityQueue[(A, Int)] = new PriorityQueue()(TmpOrd)
      val value = new BoxedArrayBuilder[(A, Int)](its.length)
      var isValid = true

      var i = 0;
      while (i < its.length) {
        if (its(i).isValid) q.enqueue(its(i).value -> i)
        i += 1
      }

      def advance() {
        var i = 0;
        while (i < value.length) {
          val j = value(i)._2
          its(j).advance()
          if (its(j).isValid) q.enqueue(its(j).value -> j)
          i += 1
        }
        value.clear()
        if (q.isEmpty) {
          isValid = false
        } else {
          val v = q.dequeue()
          value += v
          while (!q.isEmpty && ord(q.head._1, v._1) == 0)
            value += q.dequeue()
        }
      }
    }

    sm.advance()
    FlipbookIterator(sm)
  }
}

/** The primary public interface of FlipbookIterator[A] consists of the methods
  *   - isValid: Boolean
  *   - value: A
  *   - advance(): Unit
  *
  * It also extends BufferedIterator[A] for interoperability with Scala and Spark.
  *
  * To define a new FlipbookIterator, define a StateMachine (which has the same abstract methods as
  * FlipbookIterator, but is unchecked), then use the factory method FlipbookIterator(sm).
  */
abstract class FlipbookIterator[A] extends BufferedIterator[A] { self =>
  def isValid: Boolean
  def value: A
  def advance(): Unit

  def valueOrElse(default: A): A =
    if (isValid) value else default

  def exhaust() { while (isValid) advance() }

  def toStagingIterator: StagingIterator[A]

  override def filter(pred: A => Boolean): FlipbookIterator[A] = FlipbookIterator(
    new StateMachine[A] {
      def value = self.value
      def isValid = self.isValid

      def advance() {
        do self.advance() while (self.isValid && !pred(self.value))
      }

      while (self.isValid && !pred(self.value)) self.advance()
    }
  )

  override def withFilter(pred: A => Boolean): FlipbookIterator[A] = filter(pred)

  override def map[B](f: A => B): FlipbookIterator[B] = FlipbookIterator(
    new StateMachine[B] {
      var value: B = _
      if (self.isValid) value = f(self.value)
      def isValid = self.isValid

      def advance() {
        self.advance()
        if (self.isValid) value = f(self.value)
      }
    }
  )

  override def flatMap[B](f: A => GenTraversableOnce[B]): FlipbookIterator[B] =
    FlipbookIterator(
      new StateMachine[B] {
        var it: FlipbookIterator[B] = _
        if (self.isValid) it = f(self.value).toIterator.toFlipbookIterator
        findNextValid
        def value: B = it.value
        def isValid = self.isValid

        def advance() {
          it.advance()
          findNextValid
        }

        def findNextValid() {
          while (self.isValid && !it.isValid) {
            self.advance()
            if (self.isValid) it = f(self.value).toIterator.toFlipbookIterator
          }
        }
      }
    )

  private[this] trait ValidityCachingStateMachine extends StateMachine[A] {
    private[this] var _isValid: Boolean = _
    final def isValid = _isValid

    final def refreshValidity(): Unit =
      _isValid = calculateValidity

    def calculateValidity: Boolean
    def value: A
    def advance(): Unit
    refreshValidity
  }

  def staircased(ord: OrderingView[A]): StagingIterator[FlipbookIterator[A]] = {
    ord.setBottom()
    val stepSM = new ValidityCachingStateMachine {
      def value: A = self.value
      def calculateValidity: Boolean = self.isValid && ord.isEquivalent(self.value)
      def advance() = {
        self.advance()
        refreshValidity
      }
    }
    val stepIterator: FlipbookIterator[A] = FlipbookIterator(stepSM)
    val sm = new StateMachine[FlipbookIterator[A]] {
      var isValid: Boolean = true
      val value: FlipbookIterator[A] = stepIterator
      def advance() = {
        stepIterator.exhaust()
        if (self.isValid) {
          ord.setValue(self.value)
          stepSM.refreshValidity
        } else {
          ord.setBottom()
          stepSM.refreshValidity
          isValid = false
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
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[FlipbookIterator[A], FlipbookIterator[B]]] = {
    this.staircased(leftOrd).orderedZipJoin(
      that.staircased(rightOrd),
      FlipbookIterator.empty,
      FlipbookIterator.empty,
      (l, r) => mixedOrd(l.head, r.head),
    )
  }

  def orderedZipJoin[B](
    that: FlipbookIterator[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val left = self.toStagingIterator
    val right = that.toStagingIterator
    val sm = new StateMachine[Muple[A, B]] {
      val value = Muple(leftDefault, rightDefault)
      var isValid = true
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
            isValid = false
            return
          }
        }
        if (c == 0)
          value.set(left.consume(), right.consume())
        else if (c < 0)
          value.set(left.consume(), rightDefault)
        else // c > 0
          value.set(leftDefault, right.consume())
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
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    for {
      Muple(l, r) <- this.cogroup(that, leftOrd, rightOrd, mixedOrd) if r.isValid
      lrv <- l
    } yield result.set(lrv, r.value)
  }

  def leftJoinDistinct[B](
    that: FlipbookIterator[B],
    leftDefault: A,
    rightDefault: B,
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val left = self
    val right = that
    val sm = new StateMachine[Muple[A, B]] {
      val value = Muple(leftDefault, rightDefault)
      var isValid = true
      def setValue() {
        if (!left.isValid)
          isValid = false
        else {
          var c = 0
          while (right.isValid && { c = mixedOrd(left.value, right.value); c > 0 })
            right.advance()
          if (!right.isValid || c < 0)
            value.set(left.value, rightDefault)
          else // c == 0
            value.set(left.value, right.value)
        }
      }
      def advance() {
        left.advance()
        setValue()
      }

      setValue()
    }

    FlipbookIterator(sm)
  }

  def innerJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      lIt.cartesianProduct(rIt, rightBuffer, result)
    }
  }

  def leftJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (rIt.isValid) lIt.cartesianProduct(rIt, rightBuffer, result)
      else lIt.map(lElem => result.set(lElem, rightDefault))
    }
  }

  def rightJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (lIt.isValid) lIt.cartesianProduct(rIt, rightBuffer, result)
      else rIt.map(rElem => result.set(leftDefault, rElem))
    }
  }

  def outerJoin[B](
    that: FlipbookIterator[B],
    leftOrd: OrderingView[A],
    rightOrd: OrderingView[B],
    leftDefault: A,
    rightDefault: B,
    rightBuffer: Growable[B] with Iterable[B],
    mixedOrd: (A, B) => Int,
  ): FlipbookIterator[Muple[A, B]] = {
    val result = Muple[A, B](leftDefault, rightDefault)
    this.cogroup(that, leftOrd, rightOrd, mixedOrd).flatMap { case Muple(lIt, rIt) =>
      if (!lIt.isValid) rIt.map(rElem => result.set(leftDefault, rElem))
      else if (!rIt.isValid) lIt.map(lElem => result.set(lElem, rightDefault))
      else lIt.cartesianProduct(rIt, rightBuffer, result)
    }
  }

  def cartesianProduct[B](
    that: FlipbookIterator[B],
    buffer: Growable[B] with Iterable[B],
    result: Muple[A, B],
  ): FlipbookIterator[Muple[A, B]] = {
    buffer.clear()
    if (this.isValid) buffer ++= that // avoid copying right iterator when not needed
    this.flatMap(lElem =>
      buffer.iterator.toFlipbookIterator.map(rElem =>
        result.set(lElem, rElem)
      )
    )
  }

  def merge(
    that: FlipbookIterator[A],
    ord: (A, A) => Int,
  ): FlipbookIterator[A] = {
    val left = self.toStagingIterator
    val right = that.toStagingIterator
    class MergeStateMachine extends StateMachine[A] {
      var value: A = _
      var isValid = true
      def advance() {
        left.stage()
        right.stage()
        val c = {
          if (left.isValid) {
            if (right.isValid)
              ord(left.value, right.value)
            else
              -1
          } else if (right.isValid)
            1
          else {
            isValid = false
            return
          }
        }
        if (c <= 0)
          value = left.consume()
        else
          value = right.consume()
      }
    }
    val sm = new MergeStateMachine
    sm.advance()
    FlipbookIterator(sm)
  }

  def sameElementsUsing[B](that: Iterator[B], eq: (A, B) => Boolean): Boolean = {
    while (this.hasNext && that.hasNext)
      if (!eq(this.next(), that.next()))
        return false
    !this.hasNext && !that.hasNext
  }

  // head, next, and hasNext are not meant to be used directly, only to enable
  // FlipbookIterator to be used where an Iterator is expected.
  def head: A
  def next(): A
  def hasNext: Boolean
}
