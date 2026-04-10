package is.hail.collection

import is.hail.scalacheck._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.scalacheck.Arbitrary._
import org.scalacheck.Gen
import org.scalacheck.Gen._
import org.scalacheck.Prop.forAll

class BinaryHeapSuite extends munit.ScalaCheckSuite {
  test("insertOneIsMax") {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 10)
    assertEquals(bh.max(), 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.size, 1)
    assert(bh.contains(1))
    assertEquals(bh.extractMax(), 1)
    assert(!bh.contains(1))
    assertEquals(bh.size, 0)

    intercept[Exception](bh.max()): Unit
    intercept[Exception](bh.extractMax()): Unit
  }

  test("twoElements") {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 5)
    assert(bh.contains(1))
    bh.insert(2, 10)
    assert(bh.contains(2))
    assertEquals(bh.max(), 2)
    assertEquals(bh.max(), 2)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), 2)
    assert(!bh.contains(2))
    assertEquals(bh.size, 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), 1)
    assert(!bh.contains(1))
    assertEquals(bh.size, 0)
  }

  test("threeElements") {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, -10)
    assert(bh.contains(1))
    bh.insert(2, -5)
    assert(bh.contains(2))
    bh.insert(3, -7)
    assert(bh.contains(3))
    assertEquals(bh.max(), 2)
    assertEquals(bh.max(), 2)
    assertEquals(bh.size, 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.size, 2)

    assertEquals(bh.max(), 3)
    assertEquals(bh.max(), 3)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.size, 1)

    assertEquals(bh.max(), 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.size, 0)

    assert(!bh.contains(1))
    assert(!bh.contains(2))
    assert(!bh.contains(3))
  }

  test("decreaseToKey1") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(2, -10)
    assertEquals(bh.max(), 3)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.extractMax(), 2)
  }

  test("decreaseToKey2") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(1, -10)
    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
  }

  test("decreaseToKeyButNoOrderingChange") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriorityTo(3, 1)
    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
  }

  test("decreaseKey1") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(2, _ - 110)
    assertEquals(bh.max(), 3)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.extractMax(), 2)
  }

  test("decreaseKey2") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(1, _ - 10)
    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
  }

  test("decreaseKeyButNoOrderingChange") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.decreasePriority(3, _ - 9)
    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 1)
  }

  test("increaseToKey1") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriorityTo(3, 200)
    assertEquals(bh.max(), 3)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 1)
  }

  test("increaseToKeys") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriorityTo(3, 200)
    bh.increasePriorityTo(2, 300)
    bh.increasePriorityTo(1, 250)

    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.extractMax(), 3)
  }

  test("increaseKey1") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriority(3, _ + 190)
    assertEquals(bh.max(), 3)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 1)
  }

  test("increaseKeys") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.increasePriority(3, _ + 190)
    bh.increasePriority(2, _ + 200)
    bh.increasePriority(1, _ + 250)

    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.extractMax(), 3)
  }

  test("samePriority") {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 0)

    assertEquals(bh.max(), 2)
    assertEquals(bh.extractMax(), 2)

    val pair = (bh.extractMax(), bh.extractMax())
    assert(pair == ((1, 3)) || pair == ((3, 1)))
  }

  test("successivelyMoreInserts") {
    Seq(2, 4, 8, 16, 32).foreach { count =>
      val bh = new BinaryHeap[Int](8)
      val trace = ArrayBuffer.empty[String]
      trace += bh.toString()
      bh.checkHeapProperty()

      for (i <- 0 until count) {
        bh.insert(i, i.toLong)
        trace += bh.toString()
        bh.checkHeapProperty()
      }
      assertEquals(bh.size, count)
      assertEquals(bh.max(), count - 1)

      (0 until count).foreach { i =>
        val actual = bh.extractMax()
        trace += bh.toString()
        bh.checkHeapProperty()
        val expected = count - i - 1
        assertEquals(
          actual,
          expected,
          s"[$count] $actual did not equal $expected, heap: $bh; trace ${trace.mkString("\n")}",
        )
      }

      assert(bh.isEmpty)
    }
  }

  test("growPastCapacity4") {
    val bh = new BinaryHeap[Int](4)
    bh.insert(1, 0)
    bh.insert(2, 0)
    bh.insert(3, 0)
    bh.insert(4, 0)
    bh.insert(5, 0)
  }

  test("growPastCapacity32") {
    val bh = new BinaryHeap[Int](32)
    for (i <- 0 to 32)
      bh.insert(i, 0)
  }

  test("shrinkCapacity") {
    val bh = new BinaryHeap[Int](8)
    val trace = ArrayBuffer.empty[String]
    trace += bh.toString()
    bh.checkHeapProperty()
    (0 until 64).foreach { i =>
      bh.insert(i, i.toLong)
      trace += bh.toString()
      bh.checkHeapProperty()
    }
    assertEquals(bh.size, 64, s"trace: ${trace.mkString("\n")}")
    assertEquals(bh.max(), 63, s"trace: ${trace.mkString("\n")}")
    // shrinking happens when size is <1/4 of capacity
    (0 until (32 + 16 + 1)).foreach { i =>
      val actual = bh.extractMax()
      val expected = 64 - i - 1
      trace += bh.toString()
      bh.checkHeapProperty()
      assertEquals(
        actual,
        expected,
        s"$actual did not equal $expected, trace: ${trace.mkString("\n")}",
      )
    }
    assertEquals(bh.size, 15, s"trace: ${trace.mkString("\n")}")
    assertEquals(bh.max(), 14, s"trace: ${trace.mkString("\n")}")
  }

  sealed trait HeapOp

  sealed case class Max() extends HeapOp

  sealed case class ExtractMax() extends HeapOp

  sealed case class Insert(t: Long, rank: Long) extends HeapOp

  class LongPriorityQueueReference {

    import Ordering.Implicits._

    val m = new mutable.HashMap[Long, Long]()

    def isEmpty =
      m.size == 0

    def size =
      m.size

    def max(): Long =
      m.toSeq.sortWith(_ > _).head._1

    def extractMax(): Long = {
      val max = m.toSeq.sortWith(_ > _).head._1
      m -= max
      max
    }

    def insert(t: Long, rank: Long): Unit = m += (t -> rank)
  }

  property("sameAsReferenceImplementation") = forAll(
    for {
      maxOrExtract <- containerOfN[IndexedSeq, HeapOp](64, Gen.oneOf(Max(), ExtractMax()))
      ranks <- distinctContainerOfN[IndexedSeq, Long](64, arbitrary[Long])
      inserts = ranks.map(r => Insert(r, r))
    } yield Random.shuffle(inserts ++ maxOrExtract)
  ) { opList =>
    val bh = new BinaryHeap[Long]()
    val ref = new LongPriorityQueueReference()
    val trace = ArrayBuffer.empty[String]
    trace += bh.toString()
    opList.foreach {
      case Max() =>
        if (!(bh.isEmpty && ref.isEmpty))
          assertEquals(bh.max(), ref.max(), s"trace; ${trace.mkString("\n")}")
        trace += bh.toString()
        bh.checkHeapProperty()
      case ExtractMax() =>
        if (!(bh.isEmpty && ref.isEmpty))
          assertEquals(bh.max(), ref.max(), s"trace; ${trace.mkString("\n")}")
        trace += bh.toString()
        bh.checkHeapProperty()
      case Insert(t, rank) =>
        bh.insert(t, rank)
        ref.insert(t, rank)
        trace += bh.toString()
        bh.checkHeapProperty()
        assertEquals(bh.size, ref.size, s"trace; ${trace.mkString("\n")}")
    }
  }

  private def evensFirst(a: Int, b: Int): Double = {
    if (a % 2 == 0 && b % 2 == 1)
      1
    else if (a % 2 == 1 && b % 2 == 0)
      -1
    else
      0
  }

  test("tieBreakingDoesntChangeExistingFunctionality") {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -10)
    assert(bh.contains(1))
    bh.insert(2, -5)
    assert(bh.contains(2))
    bh.insert(3, -7)
    assert(bh.contains(3))
    assertEquals(bh.max(), 2)
    assertEquals(bh.max(), 2)
    assertEquals(bh.size, 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.size, 2)

    assertEquals(bh.max(), 3)
    assertEquals(bh.max(), 3)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.size, 1)

    assertEquals(bh.max(), 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.size, 0)

    assert(!bh.contains(1))
    assert(!bh.contains(2))
    assert(!bh.contains(3))
  }

  test("tieBreakingHappens") {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -10)
    assert(bh.contains(1))
    bh.insert(2, -5)
    assert(bh.contains(2))
    bh.insert(3, -5)
    assert(bh.contains(3))
    assertEquals(bh.max(), 2)
    assertEquals(bh.max(), 2)
    assertEquals(bh.size, 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.size, 2)

    assertEquals(bh.max(), 3)
    assertEquals(bh.max(), 3)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.size, 1)

    assertEquals(bh.max(), 1)
    assertEquals(bh.max(), 1)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.size, 0)

    assert(!bh.contains(1))
    assert(!bh.contains(2))
    assert(!bh.contains(3))
  }

  test("tieBreakingThreeWayDeterministic") {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, -5)
    assert(bh.contains(1))
    bh.insert(2, -5)
    assert(bh.contains(2))
    bh.insert(3, -5)
    assert(bh.contains(3))
    assertEquals(bh.max(), 2)
    assertEquals(bh.max(), 2)
    assertEquals(bh.size, 3)
    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.size, 2)

    val x = bh.max()
    val y = if (x == 3) 1 else 3
    assert(x == 3 || x == 1)
    assertEquals(bh.max(), x)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), x)
    assertEquals(bh.size, 1)

    assertEquals(bh.max(), y)
    assertEquals(bh.max(), y)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), y)
    assertEquals(bh.size, 0)

    assert(!bh.contains(1))
    assert(!bh.contains(2))
    assert(!bh.contains(3))
  }

  test("tieBreakingThreeWayNonDeterministic") {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(0, -5)
    assert(bh.contains(0))
    bh.insert(2, -5)
    assert(bh.contains(2))
    bh.insert(3, -5)
    assert(bh.contains(3))
    val firstMax = bh.max()
    val nextMax = if (firstMax == 2) 0 else 2
    assert(firstMax == 2 || firstMax == 0)
    assertEquals(bh.max(), firstMax)
    assertEquals(bh.size, 3)
    assertEquals(bh.extractMax(), firstMax)
    assertEquals(bh.size, 2)

    assertEquals(bh.max(), nextMax)
    assertEquals(bh.max(), nextMax)
    assertEquals(bh.size, 2)
    assertEquals(bh.extractMax(), nextMax)
    assertEquals(bh.size, 1)

    assertEquals(bh.max(), 3)
    assertEquals(bh.max(), 3)
    assertEquals(bh.size, 1)
    assertEquals(bh.extractMax(), 3)
    assertEquals(bh.size, 0)

    assert(!bh.contains(0))
    assert(!bh.contains(2))
    assert(!bh.contains(3))
  }

  test("tieBreakingAfterPriorityChange") {
    val bh = new BinaryHeap[Int](maybeTieBreaker = evensFirst)
    bh.insert(1, 15)
    bh.insert(2, 10)
    bh.insert(3, 5)
    bh.insert(4, 0)

    println(bh)
    assertEquals(bh.max(), 1)

    bh.decreasePriorityTo(1, 10)

    assertEquals(bh.max(), 2)

    bh.decreasePriorityTo(1, 5)

    assertEquals(bh.max(), 2)

    bh.increasePriorityTo(1, 10)

    assertEquals(bh.max(), 2)

    bh.increasePriorityTo(1, 15)

    assertEquals(bh.max(), 1)

    bh.decreasePriorityTo(1, 10)

    assertEquals(bh.extractMax(), 2)
    assertEquals(bh.max(), 1)

    bh.increasePriorityTo(4, 10)

    assertEquals(bh.extractMax(), 4)
    assertEquals(bh.extractMax(), 1)
    assertEquals(bh.extractMax(), 3)
    assert(bh.isEmpty)
  }

}
