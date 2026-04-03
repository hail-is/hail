package is.hail.collection

class ArrayStackSuite extends munit.FunSuite {
  test("basic operations") {
    val s = new IntArrayStack(4)
    assert(s.isEmpty)
    assert(!s.nonEmpty)
    assertEquals(s.size, 0)
    assertEquals(s.capacity, 4)

    s.push(13)
    assert(!s.isEmpty)
    assert(s.nonEmpty)
    assertEquals(s.size, 1)
    assertEquals(s.capacity, 4)
    assertEquals(s.top, 13)
    assertEquals(s(0), 13)

    s.push(0)
    s.push(-1)
    s.push(11)
    s.push(-13)
    assertEquals(s.size, 5)
    assert(s.capacity >= 5)
    assertEquals(s.top, -13)

    assertEquals(s(0), -13)
    assertEquals(s(1), 11)
    assertEquals(s(2), -1)
    assertEquals(s(3), 0)
    assertEquals(s(4), 13)

    s(2) = 39
    assertEquals(s.pop(), -13)
    assertEquals(s.top, 11)
    assertEquals(s.size, 4)

    assertEquals(s.pop(), 11)
    assertEquals(s.pop(), 39)
    assertEquals(s.size, 2)

    s.pop(): Unit
    s.pop(): Unit
    assert(s.isEmpty)
  }
}
