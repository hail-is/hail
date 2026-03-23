package is.hail.utils

class UnionFindSuite extends munit.FunSuite {
  test("emptyUnionFindHasNoSets") {
    assertEquals(new UnionFind().size, 0)
  }

  test("growingPastInitialCapacityOK") {
    val uf = new UnionFind(4)
    uf.makeSet(0)
    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    assertEquals(uf.find(0), 0)
    assertEquals(uf.find(1), 1)
    assertEquals(uf.find(2), 2)
    assertEquals(uf.find(3), 3)
    assertEquals(uf.find(4), 4)
    assertEquals(uf.size, 5)
  }

  test("simpleUnions") {
    val uf = new UnionFind()

    uf.makeSet(0)
    uf.makeSet(1)

    uf.union(0, 1)

    val (x, y) = (uf.find(0), uf.find(1))
    assertEquals(x, y)
    assert(x == 0 || x == 1)
  }

  test("nonMonotonicMakeSet") {
    val uf = new UnionFind()

    uf.makeSet(1000)
    uf.makeSet(1024)
    uf.makeSet(4097)
    uf.makeSet(4095)

    assertEquals(uf.find(1000), 1000)
    assertEquals(uf.find(1024), 1024)
    assertEquals(uf.find(4097), 4097)
    assertEquals(uf.find(4095), 4095)
    assert(!uf.sameSet(1000, 1024))
    assert(!uf.sameSet(1000, 4097))
    assert(!uf.sameSet(1000, 4095))
    assert(!uf.sameSet(1024, 4097))
    assert(!uf.sameSet(1024, 4095))
    assert(!uf.sameSet(4097, 4095))
    assertEquals(uf.size, 4)
  }

  test("multipleUnions") {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    assertEquals(uf.size, 4)

    uf.union(1, 2)

    assert(uf.sameSet(1, 2))
    assert(!uf.sameSet(1, 3))
    assert(!uf.sameSet(1, 4))
    assert(!uf.sameSet(3, 4))
    assertEquals(uf.size, 3)

    uf.union(1, 4)

    assert(uf.sameSet(1, 2))
    assert(!uf.sameSet(1, 3))
    assert(uf.sameSet(1, 4))
    assert(!uf.sameSet(3, 4))
    assertEquals(uf.size, 2)

    uf.union(2, 3)

    assert(uf.sameSet(1, 2))
    assert(uf.sameSet(1, 3))
    assert(uf.sameSet(1, 4))
    assertEquals(uf.size, 1)
  }

  test("unionsNoInterveningFinds") {
    val uf = new UnionFind()

    uf.makeSet(1)
    uf.makeSet(2)
    uf.makeSet(3)
    uf.makeSet(4)
    uf.makeSet(5)
    uf.makeSet(6)

    assertEquals(uf.size, 6)

    uf.union(1, 2)
    uf.union(1, 4)
    uf.union(5, 3)
    uf.union(2, 6)

    assertEquals(uf.size, 2)
    assert(uf.sameSet(1, 2))
    assert(uf.sameSet(1, 4))
    assert(uf.sameSet(5, 3))
    assert(uf.sameSet(1, 6))
    assert(!uf.sameSet(1, 5))
  }

  test("sameSetWorks") {
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
