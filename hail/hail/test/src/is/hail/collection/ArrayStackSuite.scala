package is.hail.collection

import is.hail.TestUtils._

import org.junit.jupiter.api.Test

class ArrayStackSuite {
  @Test def test(): Unit = {
    val s = new IntArrayStack(4)
    assert(s.isEmpty)
    assert(!s.nonEmpty)
    assertEq(s.size, 0)
    assertEq(s.capacity, 4)

    s.push(13)
    assert(!s.isEmpty)
    assert(s.nonEmpty)
    assertEq(s.size, 1)
    assertEq(s.capacity, 4)
    assertEq(s.top, 13)
    assertEq(s(0), 13)

    s.push(0)
    s.push(-1)
    s.push(11)
    s.push(-13)
    assertEq(s.size, 5)
    assert(s.capacity >= 5)
    assertEq(s.top, -13)

    assertEq(s(0), -13)
    assertEq(s(1), 11)
    assertEq(s(2), -1)
    assertEq(s(3), 0)
    assertEq(s(4), 13)

    s(2) = 39
    assertEq(s.pop(), -13)
    assertEq(s.top, 11)
    assertEq(s.size, 4)

    assertEq(s.pop(), 11)
    assertEq(s.pop(), 39)
    assertEq(s.size, 2)

    s.pop(): Unit
    s.pop(): Unit
    assert(s.isEmpty)
  }
}
