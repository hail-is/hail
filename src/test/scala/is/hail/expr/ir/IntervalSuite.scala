package is.hail.expr.ir

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test
import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.utils.Interval
import org.apache.spark.sql.Row

class IntervalSuite extends TestNGSuite {

  val tpoint1 = TTuple(TInt32())
  val tinterval1 = TInterval(tpoint1)
  val na = NA(tinterval1)

  def point(i: Int): IR = MakeTuple(Seq(I32(i)))
  def interval(start: IR, end: IR, includeStart: java.lang.Boolean, includeEnd: java.lang.Boolean): IR = {
    invoke("Interval", start, end, Literal(includeStart, TBoolean()), Literal(includeEnd, TBoolean()))
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
    assertEvalsTo(invoke("start", i1), Row(1))
    assertEvalsTo(invoke("start", i2), Row(1))
    assertEvalsTo(invoke("start", i3), null)
    assertEvalsTo(invoke("start", na), null)
  }

  @Test def end() {
    assertEvalsTo(invoke("end", i1), Row(2))
    assertEvalsTo(invoke("end", i2), null)
    assertEvalsTo(invoke("end", i3), Row(2))
    assertEvalsTo(invoke("end", na), null)
  }

  @Test def includeStart() {
    assertEvalsTo(invoke("includesStart", i1), true)
    assertEvalsTo(invoke("includesStart", i2), true)
    assertEvalsTo(invoke("includesStart", i3), true)
    assertEvalsTo(invoke("includesStart", na), null)
  }

  @Test def includeEnd() {
    assertEvalsTo(invoke("includesEnd", i1), false)
    assertEvalsTo(invoke("includesEnd", i2), false)
    assertEvalsTo(invoke("includesEnd", i3), false)
    assertEvalsTo(invoke("includesEnd", na), null)
  }

}
