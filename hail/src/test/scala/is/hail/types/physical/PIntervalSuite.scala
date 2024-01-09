package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.types.physical.stypes.concrete.{SUnreachableInterval, SUnreachableIntervalValue}
import is.hail.types.virtual.{TInt32, TInterval}
import is.hail.utils._

import org.testng.annotations.Test

class PIntervalSuite extends PhysicalTestUtils {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      copyTestExecutor(
        PCanonicalInterval(PInt64()),
        PCanonicalInterval(PInt64()),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalInterval(PInt64(true)),
        PCanonicalInterval(PInt64()),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalInterval(PInt64(true)),
        PCanonicalInterval(PInt64(true)),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalInterval(PInt64()),
        PCanonicalInterval(PInt64(true)),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalInterval(PInt64(true)),
        PCanonicalInterval(PInt64(true)),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        deepCopy = deepCopy,
        interpret = interpret,
      )
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }

  // Just makes sure we can generate code to store an unreachable interval
  @Test def storeUnreachable(): Unit = {
    val ust = SUnreachableInterval(TInterval(TInt32))
    val usv = new SUnreachableIntervalValue(ust)
    val pt = PCanonicalInterval(PInt32Required, true)

    val fb = EmitFunctionBuilder[Region, Long](ctx, "pinterval_store_unreachable")
    val codeRegion = fb.getCodeParam[Region](1)

    fb.emitWithBuilder(cb => pt.store(cb, codeRegion, usv, true))
  }
}
