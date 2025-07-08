package is.hail.utils

import is.hail.macros.void
import is.hail.scalacheck._

import scala.collection.mutable
import scala.util.Random

import org.scalacheck.Arbitrary._
import org.scalacheck.Gen
import org.scalacheck.Gen._
import org.scalatest
import org.scalatest.matchers.should.Matchers._
import org.scalatestplus.scalacheck.CheckerAsserting.assertingNatureOfAssertion
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class BinaryHeapSuite extends ScalaCheckDrivenPropertyChecks {
  @Test
  def insertOneIsMax(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 10)
    assert(bh.max() === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.contains(1) === true)
    assert(bh.extractMax() === 1)
    assert(bh.contains(1) === false)
    assert(bh.size === 0)

    assertThrows[Exception](bh.max())
    assertThrows[Exception](bh.extractMax())
  }

  @Test
  def twoElements(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 5)
    assert(bh.contains(1) == true)
    bh.insert(2, 10)
    assert(bh.contains(2) == true)
    assert(bh.max() === 2)
    assert(bh.max() === 2)
    assert(bh.size === 2)
    assert(bh.extractMax() === 2)
    assert(bh.contains(2) == false)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.contains(1) == false)
    assert(bh.size === 0)
  }

  @Test
  def threeElements(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, -10)
    assert(bh.contains(1) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -7)
    assert(bh.contains(3) == true)
    assert(bh.max() === 2)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.extractMax() === 2)
    assert(bh.size === 2)

    assert(bh.max() === 3)
    assert(bh.max() === 3)
    assert(bh.size === 2)
    assert(bh.extractMax() === 3)
    assert(bh.size === 1)

    assert(bh.max() === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)

    assert(bh.contains(1) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def decreaseToKey1(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(2, -10)
    assert(bh.max() === 3)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
    assert(bh.extractMax() === 2)
  }

  @Test
  def decreaseToKey2(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(1, -10)
    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
  }

  @Test
  def decreaseToKeyButNoOrderingChange(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(3, 1)
    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
  }

  @Test
  def decreaseKey1(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(2, _ - 110)
    assert(bh.max() === 3)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
    assert(bh.extractMax() === 2)
  }

  @Test
  def decreaseKey2(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(1, _ - 10)
    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
  }

  @Test
  def decreaseKeyButNoOrderingChange(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(3, _ - 9)
    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 1)
  }

  @Test
  def increaseToKey1(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriorityTo(3, 200)
    assert(bh.max() === 3)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 1)
  }

  @Test
  def increaseToKeys(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriorityTo(3, 200)
    bh.increasePriorityTo(2, 300)
    bh.increasePriorityTo(1, 250)

    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 1)
    assert(bh.extractMax() === 3)
  }

  @Test
  def increaseKey1(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriority(3, _ + 190)
    assert(bh.max() === 3)
    assert(bh.extractMax() === 3)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 1)
  }

  @Test
  def increaseKeys(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriority(3, _ + 190)
    bh.increasePriority(2, _ + 200)
    bh.increasePriority(1, _ + 250)

    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)
    assert(bh.extractMax() === 1)
    assert(bh.extractMax() === 3)
  }

  @Test
  def samePriority(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 0)

    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)

    (bh.extractMax(), bh.extractMax()) should (equal((1, 3)) or equal((3, 1)))
  }

  @Test
  def successivelyMoreInserts(): scalatest.Assertion =
    scalatest.Inspectors.forAll(Seq(2, 4, 8, 16, 32)) { count =>
      val bh = new BinaryHeap[Int](8)
      val trace = new BoxedArrayBuilder[String]()
      trace += bh.toString()
      bh.checkHeapProperty()

      for (i <- 0 until count) {
        bh.insert(i, i)
        trace += bh.toString()
        bh.checkHeapProperty()
      }
      assert(bh.size === count)
      assert(bh.max() === count - 1)

      scalatest.Inspectors.forAll(0 until count) { i =>
        val actual = bh.extractMax()
        trace += bh.toString()
        bh.checkHeapProperty()
        val expected = count - i - 1
        assert(
          actual === expected,
          s"[$count] $actual did not equal $expected, heap: $bh; trace ${trace.result().mkString("\n")}",
        )
      }

      assert(bh.isEmpty)
    }

  @Test
  def growPastCapacity4(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](4)
    bh.insert(1, 0)
    bh.insert(2, 0)
    bh.insert(3, 0)
    bh.insert(4, 0)
    bh.insert(5, 0)
    succeed
  }

  @Test
  def growPastCapacity32(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](32)
    for (i <- 0 to 32)
      bh.insert(i, 0)
    succeed
  }

  @Test
  def shrinkCapacity(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](8)
    val trace = new BoxedArrayBuilder[String]()
    trace += bh.toString()
    bh.checkHeapProperty()
    scalatest.Inspectors.forAll(0 until 64) { i =>
      bh.insert(i, i)
      trace += bh.toString()
      bh.checkHeapProperty()
    }
    assert(bh.size === 64, s"trace: ${trace.result().mkString("\n")}")
    assert(bh.max() === 63, s"trace: ${trace.result().mkString("\n")}")
    // shrinking happens when size is <1/4 of capacity
    scalatest.Inspectors.forAll(0 until (32 + 16 + 1)) { i =>
      val actual = bh.extractMax()
      val expected = 64 - i - 1
      trace += bh.toString()
      bh.checkHeapProperty()
      assert(
        actual === expected,
        s"$actual did not equal $expected, trace: ${trace.result().mkString("\n")}",
      )
    }
    assert(bh.size === 15, s"trace: ${trace.result().mkString("\n")}")
    assert(bh.max() === 14, s"trace: ${trace.result().mkString("\n")}")
  }

  sealed trait HeapOp

  sealed case class Max() extends HeapOp

  sealed case class ExtractMax() extends HeapOp

  sealed case class Insert(t: Long, rank: Long) extends HeapOp

  class LongPriorityQueueReference {

    import Ordering.Implicits._

    val m = new mutable.HashMap[Long, Long]()

    def isEmpty() =
      m.size == 0

    def size() =
      m.size

    def max(): Long =
      m.toSeq.sortWith(_ > _).head._1

    def extractMax(): Long = {
      val max = m.toSeq.sortWith(_ > _).head._1
      m -= max
      max
    }

    def insert(t: Long, rank: Long): Unit =
      void(m += (t -> rank))
  }

  @Test
  def sameAsReferenceImplementation(): scalatest.Assertion = {
    val ops =
      for {
        maxOrExtract <- containerOfN[IndexedSeq, HeapOp](64, Gen.oneOf(Max(), ExtractMax()))
        ranks <- distinctContainerOfN[IndexedSeq, Long](64, arbitrary[Long])
        inserts = ranks.map(r => Insert(r, r))
      } yield Random.shuffle(inserts ++ maxOrExtract)

    forAll(ops) { opList =>
      val bh = new BinaryHeap[Long]()
      val ref = new LongPriorityQueueReference()
      val trace = new BoxedArrayBuilder[String]()
      trace += bh.toString()
      scalatest.Inspectors.forAll(opList) {
        case Max() =>
          if (bh.isEmpty && ref.isEmpty)
            assert(true, s"trace; ${trace.result().mkString("\n")}")
          else
            assert(bh.max() === ref.max(), s"trace; ${trace.result().mkString("\n")}")
          trace += bh.toString()
          bh.checkHeapProperty()
          succeed
        case ExtractMax() =>
          if (bh.isEmpty && ref.isEmpty)
            assert(true, s"trace; ${trace.result().mkString("\n")}")
          else
            assert(bh.max() === ref.max(), s"trace; ${trace.result().mkString("\n")}")
          trace += bh.toString()
          bh.checkHeapProperty()
          succeed
        case Insert(t, rank) =>
          bh.insert(t, rank)
          ref.insert(t, rank)
          trace += bh.toString()
          bh.checkHeapProperty()
          assert(bh.size === ref.size, s"trace; ${trace.result().mkString("\n")}")
      }
    }
  }

  private def evensFirst(a: Int, b: Int): Int = {
    if (a % 2 == 0 && b % 2 == 1)
      1
    else if (a % 2 == 1 && b % 2 == 0)
      -1
    else
      0
  }

  @Test
  def tieBreakingDoesntChangeExistingFunctionality(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -10)
    assert(bh.contains(1) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -7)
    assert(bh.contains(3) == true)
    assert(bh.max() === 2)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.extractMax() === 2)
    assert(bh.size === 2)

    assert(bh.max() === 3)
    assert(bh.max() === 3)
    assert(bh.size === 2)
    assert(bh.extractMax() === 3)
    assert(bh.size === 1)

    assert(bh.max() === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)

    assert(bh.contains(1) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def tieBreakingHappens(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -10)
    assert(bh.contains(1) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -5)
    assert(bh.contains(3) == true)
    assert(bh.max() === 2)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.extractMax() === 2)
    assert(bh.size === 2)

    assert(bh.max() === 3)
    assert(bh.max() === 3)
    assert(bh.size === 2)
    assert(bh.extractMax() === 3)
    assert(bh.size === 1)

    assert(bh.max() === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)

    assert(bh.contains(1) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def tieBreakingThreeWayDeterministic(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -5)
    assert(bh.contains(1) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -5)
    assert(bh.contains(3) == true)
    assert(bh.max() === 2)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.extractMax() === 2)
    assert(bh.size === 2)

    val x = bh.max()
    val y = if (x == 3) 1 else 3
    assert(x === 3 || x === 1)
    assert(bh.max() === x)
    assert(bh.size === 2)
    assert(bh.extractMax() === x)
    assert(bh.size === 1)

    assert(bh.max() === y)
    assert(bh.max() === y)
    assert(bh.size === 1)
    assert(bh.extractMax() === y)
    assert(bh.size === 0)

    assert(bh.contains(1) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def tieBreakingThreeWayNonDeterministic(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(0, -5)
    assert(bh.contains(0) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -5)
    assert(bh.contains(3) == true)
    val firstMax = bh.max()
    val nextMax = if (firstMax == 2) 0 else 2
    assert(firstMax === 2 || firstMax === 0)
    assert(bh.max() === firstMax)
    assert(bh.size === 3)
    assert(bh.extractMax() === firstMax)
    assert(bh.size === 2)

    assert(bh.max() === nextMax)
    assert(bh.max() === nextMax)
    assert(bh.size === 2)
    assert(bh.extractMax() === nextMax)
    assert(bh.size === 1)

    assert(bh.max() === 3)
    assert(bh.max() === 3)
    assert(bh.size === 1)
    assert(bh.extractMax() === 3)
    assert(bh.size === 0)

    assert(bh.contains(0) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def tieBreakingAfterPriorityChange(): scalatest.Assertion = {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, 15)
    bh.insert(2, 10)
    bh.insert(3, 5)
    bh.insert(4, 0)

    println(bh)
    assert(bh.max() === 1)

    bh.decreasePriorityTo(1, 10)

    assert(bh.max() === 2)

    bh.decreasePriorityTo(1, 5)

    assert(bh.max() === 2)

    bh.increasePriorityTo(1, 10)

    assert(bh.max() === 2)

    bh.increasePriorityTo(1, 15)

    assert(bh.max() === 1)

    bh.decreasePriorityTo(1, 10)

    assert(bh.extractMax() === 2)
    assert(bh.max() === 1)

    bh.increasePriorityTo(4, 10)

    assert(bh.extractMax() === 4)
    assert(bh.extractMax() === 1)
    assert(bh.extractMax() === 3)
    assert(bh.isEmpty)
  }

}
