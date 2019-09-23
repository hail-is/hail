package is.hail.expr.ir

import is.hail.expr.types.virtual._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class StreamifySuite extends TestNGSuite {

  val Seq(x, y, z) = Seq("x", "y", "z").map(Ref(_, TInt32()))
  val a = Ref("a", TArray(TInt32()))

  def tests: Seq[(IR, IR)] = Seq(
    ArrayRange(0, 40, 2) ->
      ToArray(StreamRange(0, 40, 2)),
    ArrayMap(ArrayRange(0, 40, 2), "x", x + 1) ->
      ToArray(ArrayMap(StreamRange(0, 40, 2), "x", x + 1)),
    ArrayMap(ArrayMap(ArrayRange(0, 40, 2), "x", x + 1), "y", y * 2) ->
      ToArray(ArrayMap(ArrayMap(StreamRange(0, 40, 2), "x", x + 1), "y", y * 2)),
    ArrayFlatMap(ArrayRange(0, 40, 2), "x", ArrayMap(a, "y", x + y)) ->
      ToArray(ArrayFlatMap(StreamRange(0, 40, 2), "x", ArrayMap(ToStream(a), "y", x + y))),
    ArrayScan(a, 0, "a", "x", ArrayFold(ArrayRange(0, x, 1), 1, "z", "y", z * y)) ->
      ToArray(ArrayScan(ToStream(a), 0, "a", "x", ArrayFold(StreamRange(0, x, 1), 1, "z", "y", z * y))),
    ArrayMap(ArraySort(Let("x", 3, ArrayRange(0, x, 1)), "l", "r", true), "y", y) ->
      ToArray(ArrayMap(ToStream(ArraySort(Let("x", 3, StreamRange(0, x, 1)), "l", "r", true)), "y", y)),
    ArrayFilter(ToArray(ToSet(a)), "y", y < 2) ->
      ToArray(ArrayFilter(ToStream(ToSet(ToStream(a))), "y", y < 2)),
    ArrayMap(If(true, ArrayRange(0, 5, 1), ArrayRange(0, 10, 1)), "x", x - 1) ->
      ToArray(ArrayMap(
        ToStream(If(true, ToArray(StreamRange(0, 5, 1)), ToArray(StreamRange(0, 10, 1)))),
        "x", x - 1))
  )

  @Test def testStreamify() =
    for ((ir0, ir1) <- tests)
      assert(Streamify(ir0) == ir1)
}
