package is.hail.utils

import org.scalatest._
import Matchers._
import org.testng.annotations.Test

class BinaryHeapSuite {
  @Test
  def insertOneIsMax() {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 10)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.contains(1) === true)
    assert(bh.extractMax() === 1)
    assert(bh.contains(1) === false)
    assert(bh.size === 0)

    intercept[Exception](bh.max())
    intercept[Exception](bh.extractMax())
  }

  @Test
  def twoElements() {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 5)
    assert(bh.contains(1) == true)
    bh.insert(2, 10)
    assert(bh.contains(2) == true)
    assert(bh.max() === 2)
    assert(bh.size === 2)
    assert(bh.max() === 2)
    assert(bh.size === 2)
    assert(bh.extractMax() === 2)
    assert(bh.contains(2) == false)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.contains(1) == false)
    assert(bh.size === 0)
  }

  @Test
  def threeElements() {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, -10)
    assert(bh.contains(1) == true)
    bh.insert(2, -5)
    assert(bh.contains(2) == true)
    bh.insert(3, -7)
    assert(bh.contains(3) == true)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.max() === 2)
    assert(bh.size === 3)
    assert(bh.extractMax() === 2)
    assert(bh.size === 2)

    assert(bh.max() === 3)
    assert(bh.size === 2)
    assert(bh.max() === 3)
    assert(bh.size === 2)
    assert(bh.extractMax() === 3)
    assert(bh.size === 1)

    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)

    assert(bh.contains(1) == false)
    assert(bh.contains(2) == false)
    assert(bh.contains(3) == false)
  }

  @Test
  def decreaseKey1() {
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
  def decreaseKey2() {
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
  def decreaseKeyButNoOrderingChange() {
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
  def increaseKey1() {
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
  def increaseKeys() {
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
  def samePriority() {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 0)

    assert(bh.max() === 2)
    assert(bh.extractMax() === 2)

    (bh.extractMax(), bh.extractMax()) should (equal (1, 3) or equal (3, 1))
  }

  @Test
  def growPastCapacity4() {
    val bh = new BinaryHeap[Int](4)
    bh.insert(1, 0)
    bh.insert(2, 0)
    bh.insert(3, 0)
    bh.insert(4, 0)
    bh.insert(5, 0)
    assert(true)
  }

  @Test
  def growPastCapacity32() {
    val bh = new BinaryHeap[Int](32)
    for (i <- 0 to 32) {
      bh.insert(i, 0)
    }
    assert(true)
  }

  @Test
  def shrinkCapacity() {
    val bh = new BinaryHeap[Int](8)
    for (i <- 0 until 64) {
      bh.insert(i, i)
    }
    assert(bh.size() === 64)
    assert(bh.max() === 63)
    // shrinking happens when size is <1/4 of capacity
    for (i <- 0 until (32+16+1)) {
      assert(bh.extractMax() === (64-i-1))
    }
    assert(bh.size() === 16)
    assert(bh.max() === 15)
  }

}
