package is.hail.utils

import is.hail.expr.types.{TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowIntervalSuite extends TestNGSuite {
  val pord = TTuple(IndexedSeq(TInt32(), TInt32(), TInt32())).ordering

  @Test def testContains() {
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).contains(pord, Row(1, 1, 3)))
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).contains(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), false, true).contains(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, false).contains(pord, Row(1, 2, 4)))

    assert(Interval(Row(0, 1), Row(1, 2, 4), true, true).contains(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1), Row(1, 2, 4), false, true).contains(pord, Row(0, 1, 5)))
    assert(Interval(Row(0, 1), Row(0, 1, 4), true, true).contains(pord, Row(0, 1, 4)))
    assert(!Interval(Row(0, 1), Row(0, 1, 4), true, false).contains(pord, Row(0, 1, 4)))

    assert(Interval(Row(0, 1), Row(2, 1, 4), true, false).contains(pord, Row(1)))
    assert(!Interval(Row(0, 1), Row(2, 1, 4), true, false).contains(pord, Row(0)))
    assert(!Interval(Row(0, 1), Row(2, 1, 4), true, false).contains(pord, Row(2)))

    assert(Interval(Row(0, 1, 5, 7), Row(2, 1, 4, 5), true, false).contains(pord, Row(0, 1, 6)))
    assert(!Interval(Row(0, 1, 5, 7), Row(2, 1, 4, 5), true, false).contains(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1, 5, 7), Row(2, 1, 4, 5), false, false).contains(pord, Row(0, 1, 5)))
  }

  @Test def testAbovePosition() {
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).isAbovePosition(pord, Row(0, 1, 4)))
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), false, true).isAbovePosition(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).isAbovePosition(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, false).isAbovePosition(pord, Row(1, 2, 4)))

    assert(Interval(Row(0, 1), Row(1, 2, 4), true, true).isAbovePosition(pord, Row(0, 0, 5)))
    assert(Interval(Row(0, 1), Row(1, 2, 4), false, true).isAbovePosition(pord, Row(0, 1, 5)))
    assert(!Interval(Row(0, 1), Row(0, 1, 4), true, true).isAbovePosition(pord, Row(0, 1, 4)))

    assert(Interval(Row(0, 1, 2, 3), Row(1, 2, 3, 4), true, true).isAbovePosition(pord, Row(0, 1, 1, 4)))
    assert(!Interval(Row(0, 1, 2, 3), Row(1, 2, 3, 4), true, true).isAbovePosition(pord, Row(0, 1, 2, 2)))
  }

  @Test def testBelowPosition() {
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).isBelowPosition(pord, Row(1, 2, 5)))
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, false).isBelowPosition(pord, Row(1, 2, 4)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).isBelowPosition(pord, Row(1, 2, 4)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, false).isBelowPosition(pord, Row(0, 2, 4)))

    assert(Interval(Row(1, 1, 8), Row(1, 2), true, true).isBelowPosition(pord, Row(1, 3, 6)))
    assert(Interval(Row(1, 1, 8), Row(1, 2), false, false).isBelowPosition(pord, Row(1, 2, 5)))
    assert(!Interval(Row(1, 1, 8), Row(1, 2), true, true).isBelowPosition(pord, Row(1, 2, 5)))
  }

  @Test def testAbutts() {
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).abutts(pord,
      Interval(Row(1, 2, 4), Row(1, 3, 4), false, true)))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).abutts(pord,
      Interval(Row(1, 2, 4), Row(1, 3, 4), true, true)))

    assert(Interval(Row(0, 1), Row(1, 2), true, true).abutts(pord,
      Interval(Row(1, 2), Row(1, 3), false, true)))
    assert(!Interval(Row(0, 1), Row(1, 2), true, true).abutts(pord,
      Interval(Row(1, 2), Row(1, 3), true, true)))
  }

  @Test def testIsValid() {
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 2), false, false))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 0), false, false))
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, true))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, false))
  }

  val iTree = IntervalTree.annotationTree(pord, Array(
    (Interval(Row(1, 0), Row(4, 3), true, false), 1),
    (Interval(Row(4, 3), Row(7, 9), true, false), 2),
    (Interval(Row(7, 9), Row(10, 0), true, true), 3)))

  @Test def testIntervalTree() {
    assert(iTree.contains(pord, Row(1, 0, 1)))
    assert(iTree.contains(pord, Row(8, 9, 1)))
    assert(iTree.contains(pord, Row(8)))
    assert(iTree.contains(pord, Row(10, 0, 10)))

    assert(!iTree.contains(pord, Row(10, 1)))
    assert(!iTree.contains(pord, Row(10)))
    assert(!iTree.contains(pord, Row(1, -1, 2)))

    assert(iTree.probablyOverlaps(pord, Interval(Row(10), Row(11), true, true)))
    assert(!iTree.probablyOverlaps(pord, Interval(Row(10), Row(11), false, true)))

    assertResult(Array(1)) {
      iTree.queryOverlappingValues(pord, Interval(Row(1), Row(2), true, true))
    }
    assertResult(Array(1, 2)) {
      iTree.queryOverlappingValues(pord, Interval(Row(1), Row(4), true, true))
    }
    assertResult(Array(1, 2)) {
      iTree.queryOverlappingValues(pord, Interval(Row(2, 0, 0), Row(4, 5, 2), true, true))
    }
  }
}
