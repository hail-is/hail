package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row
import org.testng.ITestContext
import org.testng.annotations.{BeforeMethod, Test}

class IntervalSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  val tpoint1 = TTuple(TInt32)
  val tinterval1 = TInterval(tpoint1)
  val na = NA(tinterval1)

  def point(i: Int): IR = MakeTuple.ordered(FastSeq(I32(i)))

  def interval(start: IR, end: IR, includeStart: java.lang.Boolean, includeEnd: java.lang.Boolean)
    : IR =
    invoke(
      "Interval",
      TInterval(start.typ),
      start,
      end,
      Literal.coerce(TBoolean, includeStart),
      Literal.coerce(TBoolean, includeEnd),
    )

  val i1 = interval(point(1), point(2), true, false)
  val i2 = interval(point(1), NA(tpoint1), true, false)
  val i3 = interval(NA(tpoint1), point(2), true, false)
  val i4 = interval(NA(tpoint1), point(2), null, false)
  val i5 = interval(NA(tpoint1), point(2), true, null)

  @Test def constructor() {
    assertEvalsTo(i1, Interval(Row(1), Row(2), true, false))
    assertEvalsTo(i2, Interval(Row(1), null, true, false))
    assertEvalsTo(i3, Interval(null, Row(2), true, false))
    assertEvalsTo(i4, null)
    assertEvalsTo(i5, null)
  }

  @Test def start() {
    assertEvalsTo(invoke("start", tpoint1, i1), Row(1))
    assertEvalsTo(invoke("start", tpoint1, i2), Row(1))
    assertEvalsTo(invoke("start", tpoint1, i3), null)
    assertEvalsTo(invoke("start", tpoint1, na), null)
  }

  @Test def defaultValueCorrectlyStored() {
    assertEvalsTo(If(GetTupleElement(invoke("start", tpoint1, i1), 0).ceq(1), true, false), true)
    assertEvalsTo(If(GetTupleElement(invoke("end", tpoint1, i1), 0).ceq(2), true, false), true)
  }

  @Test def end() {
    assertEvalsTo(invoke("end", tpoint1, i1), Row(2))
    assertEvalsTo(invoke("end", tpoint1, i2), null)
    assertEvalsTo(invoke("end", tpoint1, i3), Row(2))
    assertEvalsTo(invoke("end", tpoint1, na), null)
  }

  @Test def includeStart() {
    assertEvalsTo(invoke("includesStart", TBoolean, i1), true)
    assertEvalsTo(invoke("includesStart", TBoolean, i2), true)
    assertEvalsTo(invoke("includesStart", TBoolean, i3), true)
    assertEvalsTo(invoke("includesStart", TBoolean, na), null)
  }

  @Test def includeEnd() {
    assertEvalsTo(invoke("includesEnd", TBoolean, i1), false)
    assertEvalsTo(invoke("includesEnd", TBoolean, i2), false)
    assertEvalsTo(invoke("includesEnd", TBoolean, i3), false)
    assertEvalsTo(invoke("includesEnd", TBoolean, na), null)
  }

  val points: IndexedSeq[Int] = 1 to 4

  val testIntervals: IndexedSeq[SetInterval] =
    FastSeq(
      SetInterval(1, 1, true, true),
      SetInterval(1, 1, true, false),
      SetInterval(1, 3, false, true),
      SetInterval(2, 3, false, false),
      SetInterval(1, 2, true, true),
      SetInterval(3, 1, true, false),
    )

  def toIRInterval(i: SetInterval): IR =
    invoke(
      "Interval",
      TInterval(TInt32),
      ErrorIDs.NO_ERROR,
      i.start,
      i.end,
      i.includesStart,
      i.includesEnd,
    )

  @Test def contains() {
    for {
      setInterval <- testIntervals
      p <- points
    } {
      val interval = toIRInterval(setInterval)
      assert(eval(invoke("contains", TBoolean, interval, p)) == setInterval.contains(p))
    }
  }

  @Test def isEmpty() {
    for (setInterval <- testIntervals) {
      val interval = toIRInterval(setInterval)
      assert(eval(
        invoke("isEmpty", TBoolean, ErrorIDs.NO_ERROR, interval)
      ) == setInterval.definitelyEmpty())
    }
  }

  @Test def overlaps() {
    for {
      setInterval1 <- testIntervals
      setInterval2 <- testIntervals
    } {
      val interval1 = toIRInterval(setInterval1)
      val interval2 = toIRInterval(setInterval2)
      assert(eval(
        invoke("overlaps", TBoolean, interval1, interval2)
      ) == setInterval1.probablyOverlaps(setInterval2))
    }
  }

  def intInterval(start: Int, end: Int, includesStart: Boolean = true, includesEnd: Boolean = false)
    : Interval =
    Interval(start, end, includesStart, includesEnd)

  @Test def testIntervalSortAndReduce() {
    val ord = TInt32.ordering(ctx.stateManager).intervalEndpointOrdering

    assert(Interval.union(Array[Interval](), ord).sameElements(Array[Interval]()))
    assert(Interval.union(Array(intInterval(0, 10)), ord)
      .sameElements(Array(intInterval(0, 10))))

    assert(Interval.union(
      Array(
        intInterval(0, 10),
        intInterval(0, 20, includesEnd = true),
        intInterval(20, 30),
        intInterval(40, 50),
      ).reverse,
      ord,
    ).toSeq == FastSeq(
      intInterval(0, 30),
      intInterval(40, 50),
    ))
  }

  @Test def testIntervalIntersection() {
    val ord = TInt32.ordering(ctx.stateManager).intervalEndpointOrdering

    val x1 = Array[Interval](
      intInterval(5, 10),
      intInterval(15, 20),
      intInterval(25, 26),
    )
    val x2 = Array[Interval](
      intInterval(0, 1),
      intInterval(5, 22),
      intInterval(23, 24),
      intInterval(24, 25),
      intInterval(25, 26),
      intInterval(26, 27),
    )

    val x3 = Array[Interval](
      intInterval(7, 19, includesEnd = true)
    )

    assert(Interval.intersection(x1, Array[Interval](), ord).isEmpty)
    assert(Interval.intersection(Array[Interval](), x2, ord).isEmpty)
    assert(Interval.intersection(x1, x2, ord).toSeq == x1.toSeq)
    assert(Interval.intersection(x1, x2, ord).toSeq == x1.toSeq)
    assert(Interval.intersection(x1, x3, ord).toSeq == FastSeq[Interval](
      intInterval(7, 10),
      intInterval(15, 19, includesEnd = true),
    ))
  }

  @Test def testsortedNonOverlappingIntervalsContain() {
    val intervals = Literal(
      TArray(TInterval(TInt32)),
      FastSeq(
        Interval(0, 1, includesStart = true, includesEnd = true),
        Interval(10, 20, includesStart = true, includesEnd = true),
        Interval(30, 32, includesStart = false, includesEnd = false),
        Interval(32, 32, includesStart = true, includesEnd = true),
      ),
    )

    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(-1)),
      false,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(7)),
      false,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(27)),
      false,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(30)),
      false,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(300)),
      false,
    )
    assertEvalsTo(invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(0)), true)
    assertEvalsTo(invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(1)), true)
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(10)),
      true,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(11)),
      true,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(31)),
      true,
    )
    assertEvalsTo(
      invoke("sortedNonOverlappingIntervalsContain", TBoolean, intervals, I32(32)),
      true,
    )
  }

  val partitionerKType = TStruct("k1" -> TInt32, "k2" -> TInt32, "k3" -> TInt32)
  var partitioner: Literal = _

  @BeforeMethod
  def setupRVDPartitioner(context: ITestContext): Unit = {
    partitioner = new RVDPartitioner(
      ctx.stateManager,
      partitionerKType,
      Array(
        Interval(Row(1, 0), Row(4, 3), true, false),
        Interval(Row(4, 3), Row(7, 9), true, false),
        Interval(Row(7, 11), Row(10, 0), true, true),
      ),
    ).partitionBoundsIRRepresentation
  }

  @Test def testsortedNonOverlappingPartitionIntervalsEqualRange() {
    def assertRange(interval: Interval, startIdx: Int, endIdx: Int) {
      val resultType = TTuple(TInt32, TInt32)
      val irInterval = Literal(
        RVDPartitioner.intervalIRRepresentation(partitionerKType),
        RVDPartitioner.intervalToIRRepresentation(interval, 3),
      )
      assertEvalsTo(
        invoke("partitionerFindIntervalRange", resultType, partitioner, irInterval),
        Row(startIdx, endIdx),
      )
    }
    assertRange(Interval(Row(3, 4, 0), Row(7, 11), true, true), 0, 3)
    assertRange(Interval(Row(3, 4), Row(7, 9), true, false), 0, 2)
    assertRange(Interval(Row(4), Row(5), true, true), 0, 2)
    assertRange(Interval(Row(4), Row(5), false, true), 1, 2)
    assertRange(Interval(Row(-1, 7), Row(0, 9), true, false), 0, 0)
  }

  @Test def testPointPartitionIntervalEndpointComparison() {
    def assertComp(
      point: IndexedSeq[Int],
      intervalEndpoint: IndexedSeq[Int],
      leansRight: Boolean,
      function: String,
      expected: Boolean,
    ) {
      val pointIR = MakeTuple.ordered(point.map(I32))
      val endpointIR = MakeTuple.ordered(FastSeq(
        MakeTuple.ordered(Array.tabulate(3)(i =>
          if (i < intervalEndpoint.length) I32(intervalEndpoint(i)) else NA(TInt32)
        )),
        I32(intervalEndpoint.length),
      ))
      val leansRightIR = if (leansRight) True() else False()
      assertEvalsTo(
        invoke(function, TBoolean, pointIR, endpointIR, leansRightIR),
        expected,
      )
    }
    def assertLT(point: IndexedSeq[Int], intervalEndpoint: IndexedSeq[Int], leansRight: Boolean) {
      assertComp(point, intervalEndpoint, leansRight, "pointLessThanPartitionIntervalRightEndpoint",
        true)
    }
    def assertNotLT(
      point: IndexedSeq[Int],
      intervalEndpoint: IndexedSeq[Int],
      leansRight: Boolean,
    ) {
      assertComp(point, intervalEndpoint, leansRight, "pointLessThanPartitionIntervalRightEndpoint",
        false)
    }
    assertLT(Array(1, 3, 2), Array(1, 3, 2), true)
    assertNotLT(Array(1, 3, 2), Array(1, 3, 2), false)
    assertLT(Array(1, 3, 2), Array(1, 3, 4), true)
    assertLT(Array(1, 3, 2), Array(1, 4, 1), false)
    assertLT(Array(1, 3, 2), Array(1, 4, 1), true)
    assertNotLT(Array(1, 3, 2), Array(1, 2, 4), false)
    assertNotLT(Array(1, 3, 2), Array(1, 2, 4), true)
    assertLT(Array(1, 3, 2), Array(1, 3, 4), false)
    assertLT(Array(1, 3, 2), Array(1, 3), true)
    assertNotLT(Array(1, 3, 2), Array(1, 3), false)
    assertLT(Array(1, 3, 2), Array(1, 4), true)
    assertLT(Array(1, 3, 2), Array(1, 4), false)
    assertLT(Array(1, 3, 2), Array(1), true)
    assertNotLT(Array(1, 3, 2), Array(1), false)
    assertLT(Array(1, 3, 2), Array(2), true)
    assertLT(Array(1, 3, 2), Array(2), false)
    assertLT(Array(1, 3, 2), Array[Int](), true)
    assertNotLT(Array(1, 3, 2), Array[Int](), false)
  }
}
