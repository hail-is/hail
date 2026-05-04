package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.ir.{Memoized => M}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.implicits.forTesting.BaseIROps
import is.hail.types.virtual.TFloat64

import org.testng.annotations.{DataProvider, Test}

class IRBuilderSuite extends HailSuite {

  @DataProvider(name = "ImpFncBuilders")
  def dataImpFncBuilders: Array[Array[Any]] =
    Array(
      Array(
        IRBuilder.scoped { b =>
          val x0 = b.memoize(I32(2))
          val x1 = b.memoize(Cast(x0, TFloat64))
          val x2 = b.memoize(x1 / F64(2))
          val x3 = b.strictMemoize(4 * Math.atan(1), name = Name("pi"))
          val x4 = b.memoize(x3 * x1)
          val x5 = b.memoize(x2 * x2)
          val x6 = b.memoize(x3 * x5)
          makestruct("radius" -> x2, "circumference" -> x4, "area" -> x6)
        },
        M.eval {
          for {
            x0 <- I32(2)
            x1 <- Cast(x0, TFloat64)
            x2 <- x1 / F64(2)
            x3 <- Name("pi") -> F64(4 * Math.atan(1))
            x4 <- x3 * x1
            x5 <- x2 * x2
            x6 <- x3 * x5
          } yield makestruct("radius" -> x2, "circumference" -> x4, "area" -> x6)
        },
      ),
      Array(
        streamAggIR(StreamRange(0, 1, 10)) { elt =>
          IRBuilder.scoped { b =>
            val x = b.memoize(elt + 0, scope = Scope.AGG)
            ApplyAggOp(Count())(x)
          }
        },
        streamAggIR(StreamRange(0, 1, 10)) { elt =>
          M.agg(for { x <- elt + 0 } yield ApplyAggOp(Count())(x))
        },
      ),
      Array(
        streamAggScanIR(StreamRange(0, 1, 10)) { elt =>
          IRBuilder.scoped { b =>
            val x = b.memoize(elt + 0, scope = Scope.SCAN)
            ApplyScanOp(Count())(x)
          }
        },
        streamAggScanIR(StreamRange(0, 1, 10)) { elt =>
          M.scan(for { x <- elt + 0 } yield ApplyScanOp(Count())(x))
        },
      ),
    )

  @Test(dataProvider = "ImpFncBuilders")
  def testMonadicEquivalence(ib: IR, fn: IR): Unit =
    assert(ib isAlphaEquiv (ctx, fn))
}
