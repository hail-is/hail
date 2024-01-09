package is.hail.utils

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir
import is.hail.expr.ir.In
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.virtual.{TBoolean, TInt32, TStruct}

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class RowIntervalSuite extends HailSuite {
  lazy val t = TStruct("a" -> TInt32, "b" -> TInt32, "c" -> TInt32)
  lazy val pord = PartitionBoundOrdering(ctx, t)

  def assertContains(i: Interval, point: Row, shouldContain: Boolean = true): Unit = {
    val c = i.contains(pord, point)
    if (shouldContain)
      assert(c)
    else
      assert(!c)

    val tt = t.truncate(point.length)
    val irRepIntervalType = RVDPartitioner.intervalIRRepresentation(tt)
    val intervalIRRep = RVDPartitioner.intervalToIRRepresentation(i.coarsen(tt.size), tt.size)

    val in1 = In(0, irRepIntervalType)
    val in2 = In(1, tt)

    assertEvalsTo(
      ir.invoke("partitionIntervalContains", TBoolean, in1, in2),
      args = FastSeq((intervalIRRep, irRepIntervalType), (point, tt)),
      shouldContain,
    )(ExecStrategy.compileOnly)
  }

  @Test def testContains() {
    assertContains(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true), Row(1, 1, 3))
    assertContains(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true), Row(0, 1, 5))
    assertContains(
      Interval(Row(0, 1, 5), Row(1, 2, 4), false, true),
      Row(0, 1, 5),
      shouldContain = false,
    )
    assertContains(
      Interval(Row(0, 1, 5), Row(1, 2, 4), true, false),
      Row(1, 2, 4),
      shouldContain = false,
    )

    assertContains(Interval(Row(0, 1), Row(1, 2, 4), true, true), Row(0, 1, 5))
    assertContains(
      Interval(Row(0, 1), Row(1, 2, 4), false, true),
      Row(0, 1, 5),
      shouldContain = false,
    )
    assertContains(Interval(Row(0, 1), Row(0, 1, 4), true, true), Row(0, 1, 4))
    assertContains(
      Interval(Row(0, 1), Row(0, 1, 4), true, false),
      Row(0, 1, 4),
      shouldContain = false,
    )

    assertContains(Interval(Row(0, 1), Row(1, 2, 4), true, true), Row(0, 1, 5))
    assertContains(
      Interval(Row(0, 1), Row(1, 2, 4), false, true),
      Row(0, 1, 5),
      shouldContain = false,
    )
    assertContains(Interval(Row(0, 1), Row(0, 1, 4), true, true), Row(0, 1, 4))
    assertContains(
      Interval(Row(0, 1), Row(0, 1, 4), true, false),
      Row(0, 1, 4),
      shouldContain = false,
    )

    assertContains(Interval(Row(), Row(1, 2, 4), true, true), Row(1, 2, 4))
    assertContains(Interval(Row(), Row(1, 2, 4), true, false), Row(1, 2, 4), shouldContain = false)
    assertContains(Interval(Row(1, 2, 4), Row(), true, true), Row(1, 2, 4))
    assertContains(Interval(Row(1, 2, 4), Row(), false, true), Row(1, 2, 4), shouldContain = false)

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

    assert(Interval(Row(0, 1, 2, 3), Row(1, 2, 3, 4), true, true).isAbovePosition(
      pord,
      Row(0, 1, 1, 4),
    ))
    assert(!Interval(Row(0, 1, 2, 3), Row(1, 2, 3, 4), true, true).isAbovePosition(
      pord,
      Row(0, 1, 2, 2),
    ))
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
    assert(Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).abutts(
      pord,
      Interval(Row(1, 2, 4), Row(1, 3, 4), false, true),
    ))
    assert(!Interval(Row(0, 1, 5), Row(1, 2, 4), true, true).abutts(
      pord,
      Interval(Row(1, 2, 4), Row(1, 3, 4), true, true),
    ))

    assert(Interval(Row(0, 1), Row(1, 2), true, true).abutts(
      pord,
      Interval(Row(1, 2), Row(1, 3), false, true),
    ))
    assert(!Interval(Row(0, 1), Row(1, 2), true, true).abutts(
      pord,
      Interval(Row(1, 2), Row(1, 3), true, true),
    ))
  }

  @Test def testLteqWithOverlap() {
    val eord = pord.intervalEndpointOrdering
    assert(!eord.lteqWithOverlap(3)(
      IntervalEndpoint(Row(0, 1, 6), -1),
      IntervalEndpoint(Row(0, 1, 5), 1),
    ))

    assert(eord.lteqWithOverlap(3)(
      IntervalEndpoint(Row(0, 1, 5), 1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 5), 1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 5), -1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 5), -1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(Row(0, 1, 2), -1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 2), -1),
      IntervalEndpoint(Row(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1), -1),
      IntervalEndpoint(Row(0, 1), -1),
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 1), -1),
      IntervalEndpoint(Row(0, 1), -1),
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(Row(0, 1, 5), -1),
      IntervalEndpoint(Row(0, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 1, 5), -1),
      IntervalEndpoint(Row(0, 2), -1),
    ))

    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0), -1),
      IntervalEndpoint(Row(0), -1),
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0), -1),
      IntervalEndpoint(Row(0, 1, 2), 1),
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(Row(0, 3), -1),
      IntervalEndpoint(Row(1, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(Row(0, 3), -1),
      IntervalEndpoint(Row(1, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(Row(), 1),
      IntervalEndpoint(Row(), -1),
    ))
  }

  @Test def testIsValid() {
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 2), false, false))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 0), false, false))
    assert(Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, true))
    assert(!Interval.isValid(pord, Row(0, 1, 5), Row(0, 1), false, false))
  }
}
