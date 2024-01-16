package is.hail.utils

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class ArrayStackSuite extends TestNGSuite {
  @Test def test(): Unit = {
    val s = new IntArrayStack(4)
    assert(s.isEmpty)
    assert(!s.nonEmpty)
    assert(s.size == 0)
    assert(s.capacity == 4)

    s.push(13)
    assert(!s.isEmpty)
    assert(s.nonEmpty)
    assert(s.size == 1)
    assert(s.capacity == 4)
    assert(s.top == 13)
    assert(s(0) == 13)

    s.push(0)
    s.push(-1)
    s.push(11)
    s.push(-13)
    assert(s.size == 5)
    assert(s.capacity >= 5)
    assert(s.top == -13)

    assert(s(0) == -13)
    assert(s(1) == 11)
    assert(s(2) == -1)
    assert(s(3) == 0)
    assert(s(4) == 13)

    s(2) = 39
    assert(s.pop() == -13)
    assert(s.top == 11)
    assert(s.size == 4)

    assert(s.pop() == 11)
    assert(s.pop() == 39)
    assert(s.size == 2)

    s.pop()
    s.pop()
    assert(s.isEmpty)
  }
}
