package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.expr.types.virtual.{TBoolean, TInt32, TInterval, TTuple}
import is.hail.utils._
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class IntervalSuite extends TestNGSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  val tpoint1 = TTuple(TInt32())
  val tinterval1 = TInterval(tpoint1)
  val na = NA(tinterval1)

  def point(i: Int): IR = MakeTuple.ordered(Seq(I32(i)))
  def interval(start: IR, end: IR, includeStart: java.lang.Boolean, includeEnd: java.lang.Boolean): IR = {
    invoke("Interval", TInterval(start.typ), start, end, Literal.coerce(TBoolean(), includeStart), Literal.coerce(TBoolean(), includeEnd))
  }

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
    assertEvalsTo(invoke("includesStart", TBoolean(), i1), true)
    assertEvalsTo(invoke("includesStart", TBoolean(), i2), true)
    assertEvalsTo(invoke("includesStart", TBoolean(), i3), true)
    assertEvalsTo(invoke("includesStart", TBoolean(), na), null)
  }

  @Test def includeEnd() {
    assertEvalsTo(invoke("includesEnd", TBoolean(), i1), false)
    assertEvalsTo(invoke("includesEnd", TBoolean(), i2), false)
    assertEvalsTo(invoke("includesEnd", TBoolean(), i3), false)
    assertEvalsTo(invoke("includesEnd", TBoolean(), na), null)
  }

  val points: IndexedSeq[Int] = 1 to 4

  val testIntervals: IndexedSeq[SetInterval] =
    FastIndexedSeq(
      SetInterval(1, 1, true, true),
      SetInterval(1, 1, true, false),
      SetInterval(1, 3, false, true),
      SetInterval(2, 3, false, false),
      SetInterval(1, 2, true, true),
      SetInterval(3, 1, true, false))

  def toIRInterval(i: SetInterval): IR =
    invoke("Interval", TInterval(TInt32()), i.start, i.end, i.includesStart, i.includesEnd)

  @Test def contains() {
    for (setInterval <- testIntervals; p <- points) {
      val interval = toIRInterval(setInterval)
      assert(eval(invoke("contains", TBoolean(), interval, p)) == setInterval.contains(p))
    }
  }

  @Test def isEmpty() {
    for (setInterval <- testIntervals) {
      val interval = toIRInterval(setInterval)
      assert(eval(invoke("isEmpty", TBoolean(), interval)) == setInterval.definitelyEmpty())
    }
  }

  @Test def overlaps() {
    for (setInterval1 <- testIntervals; setInterval2 <- testIntervals) {
      val interval1 = toIRInterval(setInterval1)
      val interval2 = toIRInterval(setInterval2)
      assert(eval(invoke("overlaps", TBoolean(), interval1, interval2)) == setInterval1.probablyOverlaps(setInterval2))
    }
  }


  def intInterval(start: Int, end: Int, includesStart: Boolean = true, includesEnd: Boolean = false): Interval =
    Interval(start, end, includesStart, includesEnd)

  @Test def testIntervalSortAndReduce() {
    val ord = TInt32().ordering.intervalEndpointOrdering

    assert(Interval.union(Array[Interval](), ord).sameElements(Array[Interval]()))
    assert(Interval.union(Array(intInterval(0, 10)), ord)
      .sameElements(Array(intInterval(0, 10))))

    assert(Interval.union(Array(
      intInterval(0, 10),
      intInterval(0, 20, includesEnd = true),
      intInterval(20, 30),
      intInterval(40, 50)
    ).reverse, ord).toSeq == FastSeq(
      intInterval(0, 30),
      intInterval(40, 50)
    ))
  }

  @Test def testIntervalIntersection() {
    val ord = TInt32().ordering.intervalEndpointOrdering

    val x1 = Array[Interval](
      intInterval(5, 10),
      intInterval(15, 20),
      intInterval(25, 26)
    )
    val x2 = Array[Interval](
      intInterval(0, 1),
      intInterval(5, 22),
      intInterval(23, 24),
      intInterval(24, 25),
      intInterval(25, 26),
      intInterval(26, 27)
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
      intInterval(15, 19, includesEnd = true)))
  }
}
