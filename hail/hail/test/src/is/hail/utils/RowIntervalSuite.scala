package is.hail.utils

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir
import is.hail.expr.ir.defs.In
import is.hail.rvd.{PartitionBoundOrdering, RVDPartitioner}
import is.hail.types.virtual.{TBoolean, TInt32, TStruct}

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class RowIntervalSuite {
  lazy val t = TStruct("a" -> TInt32, "b" -> TInt32, "c" -> TInt32)
  def pord(implicit ctx: ExecuteContext) = PartitionBoundOrdering(ctx, t)

  def assertContains(
    i: Interval,
    point: Row,
    shouldContain: Boolean = true,
  )(implicit ctx: ExecuteContext
  ): Unit = {
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
    )(implicitly, ExecStrategy.compileOnly)
  }

  @Test def testContains(implicit ctx: ExecuteContext): Unit = {
    assertContains(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true), RowSeq(1, 1, 3))
    assertContains(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true), RowSeq(0, 1, 5))
    assertContains(
      Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), false, true),
      RowSeq(0, 1, 5),
      shouldContain = false,
    )
    assertContains(
      Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, false),
      RowSeq(1, 2, 4),
      shouldContain = false,
    )

    assertContains(Interval(RowSeq(0, 1), RowSeq(1, 2, 4), true, true), RowSeq(0, 1, 5))
    assertContains(
      Interval(RowSeq(0, 1), RowSeq(1, 2, 4), false, true),
      RowSeq(0, 1, 5),
      shouldContain = false,
    )
    assertContains(Interval(RowSeq(0, 1), RowSeq(0, 1, 4), true, true), RowSeq(0, 1, 4))
    assertContains(
      Interval(RowSeq(0, 1), RowSeq(0, 1, 4), true, false),
      RowSeq(0, 1, 4),
      shouldContain = false,
    )

    assertContains(Interval(RowSeq(0, 1), RowSeq(1, 2, 4), true, true), RowSeq(0, 1, 5))
    assertContains(
      Interval(RowSeq(0, 1), RowSeq(1, 2, 4), false, true),
      RowSeq(0, 1, 5),
      shouldContain = false,
    )
    assertContains(Interval(RowSeq(0, 1), RowSeq(0, 1, 4), true, true), RowSeq(0, 1, 4))
    assertContains(
      Interval(RowSeq(0, 1), RowSeq(0, 1, 4), true, false),
      RowSeq(0, 1, 4),
      shouldContain = false,
    )

    assertContains(Interval(RowSeq(), RowSeq(1, 2, 4), true, true), RowSeq(1, 2, 4))
    assertContains(
      Interval(RowSeq(), RowSeq(1, 2, 4), true, false),
      RowSeq(1, 2, 4),
      shouldContain = false,
    )
    assertContains(Interval(RowSeq(1, 2, 4), RowSeq(), true, true), RowSeq(1, 2, 4))
    assertContains(
      Interval(RowSeq(1, 2, 4), RowSeq(), false, true),
      RowSeq(1, 2, 4),
      shouldContain = false,
    )

    assert(Interval(RowSeq(0, 1, 5, 7), RowSeq(2, 1, 4, 5), true, false).contains(
      pord,
      RowSeq(0, 1, 6),
    ))
    assert(!Interval(RowSeq(0, 1, 5, 7), RowSeq(2, 1, 4, 5), true, false).contains(
      pord,
      RowSeq(0, 1, 5),
    ))
    assert(!Interval(RowSeq(0, 1, 5, 7), RowSeq(2, 1, 4, 5), false, false).contains(
      pord,
      RowSeq(0, 1, 5),
    ))
  }

  @Test def testAbovePosition(implicit ctx: ExecuteContext): Unit = {
    assert(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 4),
    ))
    assert(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), false, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 5),
    ))
    assert(!Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 5),
    ))
    assert(!Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, false).isAbovePosition(
      pord,
      RowSeq(1, 2, 4),
    ))

    assert(Interval(RowSeq(0, 1), RowSeq(1, 2, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 0, 5),
    ))
    assert(Interval(RowSeq(0, 1), RowSeq(1, 2, 4), false, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 5),
    ))
    assert(!Interval(RowSeq(0, 1), RowSeq(0, 1, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 4),
    ))

    assert(Interval(RowSeq(0, 1, 2, 3), RowSeq(1, 2, 3, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 1, 4),
    ))
    assert(!Interval(RowSeq(0, 1, 2, 3), RowSeq(1, 2, 3, 4), true, true).isAbovePosition(
      pord,
      RowSeq(0, 1, 2, 2),
    ))
  }

  @Test def testBelowPosition(implicit ctx: ExecuteContext): Unit = {
    assert(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).isBelowPosition(
      pord,
      RowSeq(1, 2, 5),
    ))
    assert(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, false).isBelowPosition(
      pord,
      RowSeq(1, 2, 4),
    ))
    assert(!Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).isBelowPosition(
      pord,
      RowSeq(1, 2, 4),
    ))
    assert(!Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, false).isBelowPosition(
      pord,
      RowSeq(0, 2, 4),
    ))

    assert(Interval(RowSeq(1, 1, 8), RowSeq(1, 2), true, true).isBelowPosition(
      pord,
      RowSeq(1, 3, 6),
    ))
    assert(Interval(RowSeq(1, 1, 8), RowSeq(1, 2), false, false).isBelowPosition(
      pord,
      RowSeq(1, 2, 5),
    ))
    assert(!Interval(RowSeq(1, 1, 8), RowSeq(1, 2), true, true).isBelowPosition(
      pord,
      RowSeq(1, 2, 5),
    ))
  }

  @Test def testAbutts(implicit ctx: ExecuteContext): Unit = {
    assert(Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).abutts(
      pord,
      Interval(RowSeq(1, 2, 4), RowSeq(1, 3, 4), false, true),
    ))
    assert(!Interval(RowSeq(0, 1, 5), RowSeq(1, 2, 4), true, true).abutts(
      pord,
      Interval(RowSeq(1, 2, 4), RowSeq(1, 3, 4), true, true),
    ))

    assert(Interval(RowSeq(0, 1), RowSeq(1, 2), true, true).abutts(
      pord,
      Interval(RowSeq(1, 2), RowSeq(1, 3), false, true),
    ))
    assert(!Interval(RowSeq(0, 1), RowSeq(1, 2), true, true).abutts(
      pord,
      Interval(RowSeq(1, 2), RowSeq(1, 3), true, true),
    ))
  }

  @Test def testLteqWithOverlap(implicit ctx: ExecuteContext): Unit = {
    val eord = pord.intervalEndpointOrdering
    assert(!eord.lteqWithOverlap(3)(
      IntervalEndpoint(RowSeq(0, 1, 6), -1),
      IntervalEndpoint(RowSeq(0, 1, 5), 1),
    ))

    assert(eord.lteqWithOverlap(3)(
      IntervalEndpoint(RowSeq(0, 1, 5), 1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(2)(
      IntervalEndpoint(RowSeq(0, 1, 5), 1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(2)(
      IntervalEndpoint(RowSeq(0, 1, 2), -1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))
    assert(!eord.lteqWithOverlap(1)(
      IntervalEndpoint(RowSeq(0, 1, 2), -1),
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(RowSeq(0, 1), -1),
      IntervalEndpoint(RowSeq(0, 1), -1),
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(RowSeq(0, 1), -1),
      IntervalEndpoint(RowSeq(0, 1), -1),
    ))

    assert(eord.lteqWithOverlap(1)(
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
      IntervalEndpoint(RowSeq(0, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(0)(
      IntervalEndpoint(RowSeq(0, 1, 5), -1),
      IntervalEndpoint(RowSeq(0, 2), -1),
    ))

    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(RowSeq(0), -1),
      IntervalEndpoint(RowSeq(0), -1),
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(RowSeq(0), -1),
      IntervalEndpoint(RowSeq(0, 1, 2), 1),
    ))
    assert(eord.lteqWithOverlap(0)(
      IntervalEndpoint(RowSeq(0, 3), -1),
      IntervalEndpoint(RowSeq(1, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(RowSeq(0, 3), -1),
      IntervalEndpoint(RowSeq(1, 2), -1),
    ))
    assert(!eord.lteqWithOverlap(-1)(
      IntervalEndpoint(RowSeq(), 1),
      IntervalEndpoint(RowSeq(), -1),
    ))
  }

  @Test def testIsValid(implicit ctx: ExecuteContext): Unit = {
    assert(Interval.isValid(pord, RowSeq(0, 1, 5), RowSeq(0, 2), false, false))
    assert(!Interval.isValid(pord, RowSeq(0, 1, 5), RowSeq(0, 0), false, false))
    assert(Interval.isValid(pord, RowSeq(0, 1, 5), RowSeq(0, 1), false, true))
    assert(!Interval.isValid(pord, RowSeq(0, 1, 5), RowSeq(0, 1), false, false))
  }
}
