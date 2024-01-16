package is.hail.utils

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class UnionFindSuite extends TestNGSuite {
  @Test
  def emptyUnionFindHasNoSets(): Unit =
    assert(new UnionFind().size == 0)

  @Test
  def growingPastInitialCapacityOK(): Unit = {
    val uf = new UnionFind(4)
    uf.makeSet(0)
    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    assert(uf.find(0) == 0)
    assert(uf.find(1) == 1)
    assert(uf.find(2) == 2)
    assert(uf.find(3) == 3)
    assert(uf.find(4) == 4)
    assert(uf.size == 5)
  }

  @Test
  def simpleUnions(): Unit = {
    val uf = new UnionFind()

    uf.makeSet(0)
    uf.makeSet(1)

    uf.union(0, 1)

    val (x, y) = (uf.find(0), uf.find(1))
    assert(x == y)
    assert(x == 0 || x == 1)
  }

  @Test
  def nonMonotonicMakeSet(): Unit = {
    val uf = new UnionFind()

    uf.makeSet(1000)
    uf.makeSet(1024)
    uf.makeSet(4097)
    uf.makeSet(4095)

    assert(uf.find(1000) == 1000)
    assert(uf.find(1024) == 1024)
    assert(uf.find(4097) == 4097)
    assert(uf.find(4095) == 4095)
    assert(!uf.sameSet(1000, 1024))
    assert(!uf.sameSet(1000, 4097))
    assert(!uf.sameSet(1000, 4095))
    assert(!uf.sameSet(1024, 4097))
    assert(!uf.sameSet(1024, 4095))
    assert(!uf.sameSet(4097, 4095))
    assert(uf.size == 4)
  }

  @Test
  def multipleUnions(): Unit = {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    assert(uf.size == 4)

    uf.union(1, 2)

    assert(uf.sameSet(1, 2))
    assert(!uf.sameSet(1, 3))
    assert(!uf.sameSet(1, 4))
    assert(!uf.sameSet(3, 4))
    assert(uf.size == 3)

    uf.union(1, 4)

    assert(uf.sameSet(1, 2))
    assert(!uf.sameSet(1, 3))
    assert(uf.sameSet(1, 4))
    assert(!uf.sameSet(3, 4))
    assert(uf.size == 2)

    uf.union(2, 3)

    assert(uf.sameSet(1, 2))
    assert(uf.sameSet(1, 3))
    assert(uf.sameSet(1, 4))
    assert(uf.size == 1)
  }

  @Test
  def unionsNoInterveningFinds(): Unit = {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    uf.makeSet(5)
    uf.makeSet(6)

    assert(uf.size == 6)

    uf.union(1, 2)
    uf.union(1, 4)
    uf.union(5, 3)
    uf.union(2, 6)

    assert(uf.size == 2)
    assert(uf.sameSet(1, 2))
    assert(uf.sameSet(1, 4))
    assert(uf.sameSet(5, 3))
    assert(uf.sameSet(1, 6))
    assert(!uf.sameSet(1, 5))
  }

  @Test
  def sameSetWorks(): Unit = {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(1024)
    uf.makeSet(4097)
    uf.makeSet(4096)

    assert(!uf.sameSet(1, 1024))
    assert(!uf.sameSet(1, 4097))
    assert(!uf.sameSet(1, 4096))
    assert(!uf.sameSet(2, 1024))
    assert(!uf.sameSet(2, 4097))
    assert(!uf.sameSet(2, 4096))

    uf.union(1024, 4096)
    uf.union(4097, 1)

    assert(!uf.sameSet(1, 1024))
    assert(uf.sameSet(1, 4097))
    assert(!uf.sameSet(1, 4096))
    assert(!uf.sameSet(2, 1024))
    assert(!uf.sameSet(2, 4097))
    assert(!uf.sameSet(2, 4096))

    assert(uf.sameSet(1024, 4096))
  }
}
