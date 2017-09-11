package is.hail.utils

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class UnionFindSuite extends TestNGSuite {
  @Test
  def emptyUnionFindHasNoSets() {
    assert(new UnionFind().size == 0)
  }

  @Test
  def growingPastInitialCapacityOK() {
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
  def simpleUnions() {
    val uf = new UnionFind()

    uf.makeSet(0)
    uf.makeSet(1)

    uf.union(0, 1)

    val (x, y) = (uf.find(0), uf.find(1))
    assert(x == y)
    assert(x == 0 || x == 1)
  }

  @Test
  def nonMonotonicMakeSet() {
    val uf = new UnionFind()

    uf.makeSet(1000)
    uf.makeSet(1024)
    uf.makeSet(4097)
    uf.makeSet(4095)

    assert(uf.find(1000) == 1000)
    assert(uf.find(1024) == 1024)
    assert(uf.find(4097) == 4097)
    assert(uf.find(4096) == 4096)
    assert(uf.size == 4)
  }

  @Test
  def multipleUnions() {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    assert(uf.size == 4)

    uf.union(1, 2)

    assert(uf.find(1) == uf.find(2))
    assert(uf.find(1) != uf.find(3))
    assert(uf.find(1) != uf.find(4))
    assert(uf.find(3) != uf.find(4))
    assert(uf.size == 3)

    uf.union(1, 4)

    assert(uf.find(1) == uf.find(2))
    assert(uf.find(1) != uf.find(3))
    assert(uf.find(1) == uf.find(4))
    assert(uf.find(3) != uf.find(4))
    assert(uf.size == 2)

    uf.union(2, 3)

    assert(uf.find(1) == uf.find(2))
    assert(uf.find(1) == uf.find(3))
    assert(uf.find(1) == uf.find(4))
    assert(uf.size == 1)
  }

  @Test
  def unionsNoInterveningFinds() {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    uf.makeSet(5)
    uf.makeSet(6)

    assert(uf.size == 5)

    uf.union(1, 2)
    uf.union(1, 4)
    uf.union(5, 3)
    uf.union(2, 6)

    assert(uf.size == 3)
    assert(uf.find(1) == uf.find(2))
    assert(uf.find(1) == uf.find(4))
    assert(uf.find(5) == uf.find(3))
    assert(uf.find(1) == uf.find(6))
    assert(uf.find(1) != uf.find(5))
  }
}
