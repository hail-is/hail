package is.hail.utils

import is.hail.expr.types.virtual.{TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class RowIntervalSuite extends TestNGSuite {
  val pord = TTuple(TInt32, TInt32, TInt32).ordering

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

  @Test def testLteqWithOverlap() {
    val eord = pord.intervalEndpointOrdering
    assert(!eord.lteqWithOverlap(3)(
      IntervalEndpoint(Row(0, 1, 6), -1), IntervalEndpoint(Row(0, 1, 5), 1)
    ))

    assert(eord.lteqWithOverlap(3)(
      IntervalEndpoint(Row(0, 1, 5), 1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))
    assert(!eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 5), 1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 5), -1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 5), -1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 2), -1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 2), -1), IntervalEndpoint(Row(0, 1, 5), -1)
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1), -1), IntervalEndpoint(Row(0, 1), -1)
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 1), -1), IntervalEndpoint(Row(0, 1), -1)
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 5), -1), IntervalEndpoint(Row(0, 2), -1)
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 1, 5), -1), IntervalEndpoint(Row(0, 2), -1)
    ))

    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(0), -1)
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(0, 1, 2), 1)
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 3), -1), IntervalEndpoint(Row(1, 2), -1)
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(Row(0, 3), -1), IntervalEndpoint(Row(1, 2), -1)
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(Row(), 1), IntervalEndpoint(Row(), -1)
    ))
  }

  @Test def testIsValid() {
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 2), false, false))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 0), false, false))
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, true))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, false))
  }
}
