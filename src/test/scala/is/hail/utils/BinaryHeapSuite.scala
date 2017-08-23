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
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)
    intercept[Exception](bh.max())
    intercept[Exception](bh.extractMax())
  }

  @Test
  def twoElements() {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, 5)
    bh.insert(2, 10)
    assert(bh.max() === 2)
    assert(bh.size === 2)
    assert(bh.max() === 2)
    assert(bh.size === 2)
    assert(bh.extractMax() === 2)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.max() === 1)
    assert(bh.size === 1)
    assert(bh.extractMax() === 1)
    assert(bh.size === 0)
  }

  @Test
  def threeElements() {
    val bh = new BinaryHeap[Int]()
    bh.insert(1, -10)
    bh.insert(2, -5)
    bh.insert(3, -7)
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
  }

  @Test
  def decreaseKey1() {
    val bh = new BinaryHeap[Int]()

    bh.insert(1, 0)
    bh.insert(2, 100)
    bh.insert(3, 10)

    bh.setPriority(2, -10)
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

    bh.setPriority(1, -10)
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

    bh.setPriority(3, 1)
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

    bh.setPriority(3, 200)
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

    bh.setPriority(3, 200)
    bh.setPriority(2, 300)
    bh.setPriority(1, 250)

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


}
