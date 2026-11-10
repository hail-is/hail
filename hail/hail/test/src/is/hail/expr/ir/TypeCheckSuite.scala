package is.hail.expr.ir

import is.hail.ParameterizedTest
import is.hail.TestUtils.interceptException
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._
import is.hail.types.virtual.{TArray, TInt32, TInt64}

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test

class TypeCheckSuite {
  def aggScopes = ArraySeq(Scope.AGG, Scope.SCAN)

  @ParameterizedTest("aggScopes")
  def testValueCannotSeeEvalBinding(scope: Scope)(implicit ctx: ExecuteContext): Unit = {
    val badIr =
      IRBuilder.scoped { ib =>
        val x = ib.strictMemoize(0, Name("x"))
        val y = ib.strictMemoize(x, Name("y"), scope)
        AggFilter(y ceq 0, x, scope == Scope.SCAN)
      }

    interceptException[NoSuchElementException]("x could not be resolved") {
      TypeCheck(ctx, badIr, BindingEnv.empty.createAgg.createScan)
    }
  }

  @ParameterizedTest("aggScopes")
  def testValueSeesBindingInSameScope(scope: Scope)(implicit ctx: ExecuteContext): Unit = {
    val goodIr =
      IRBuilder.scoped { ib =>
        val x = ib.strictMemoize(0, Name("x"))
        val y = ib.strictMemoize(0, Name("y"), scope)
        val z = ib.strictMemoize(y, Name("z"), scope)
        AggFilter(z ceq 0, x, scope == Scope.SCAN)
      }

    TypeCheck(ctx, goodIr, BindingEnv.empty.createAgg.createScan)
  }

  @Test def testBufferedAggregateInitAggsCannotSeeElement(implicit ctx: ExecuteContext): Unit = {
    val x = Ref(Name("x"), TInt32)
    val badIr = StreamBufferedAggregate(
      rangeIR(I32(10)),
      bindIR(x)(_ => Void()),
      MakeStruct(ArraySeq("x" -> x)),
      Void(),
      x.name,
      ArraySeq(),
      8,
    )

    interceptException[NoSuchElementException]("x could not be resolved") {
      TypeCheck(ctx, badIr)
    }
  }

  // Promoting scan drops agg: an agg op can't sit inside a scan.
  def aggsInsideScans = {
    def agg = ApplyAggOp(Count())()
    def scan = ApplyScanOp(Count())()
    ArraySeq(
      ApplyScanOp(Sum())(agg),
      Block(
        ArraySeq(Binding(Name("x"), agg, Scope.SCAN)),
        ApplyScanOp(Sum())(Ref(Name("x"), TInt64)),
      ),
      AggFilter(agg ceq I64(0), scan, true),
      AggGroupBy(agg, scan, true),
      AggExplode(ToStream(MakeArray(ArraySeq(agg), TArray(TInt64))), Name("e"), scan, true),
      AggArrayPerElement(
        MakeArray(ArraySeq(agg), TArray(TInt64)),
        Name("e"),
        Name("i"),
        scan,
        None,
        true,
      ),
      aggFoldIR(I64(0), isScan = true)(acc => acc + agg)(_ + _),
    )
  }

  @ParameterizedTest("aggsInsideScans")
  def testAggInsideScanFails(badIr: IR)(implicit ctx: ExecuteContext): Unit =
    interceptException[AssertionError]("ApplyAggOp outside an agg scope") {
      TypeCheck(ctx, badIr, BindingEnv.empty.createAgg.createScan)
    }

  // Promoting agg drops scan: a scan op can't sit inside an agg.
  def scansInsideAggs = {
    def agg = ApplyAggOp(Count())()
    def scan = ApplyScanOp(Count())()
    ArraySeq(
      ApplyAggOp(Sum())(scan),
      Block(
        ArraySeq(Binding(Name("x"), scan, Scope.AGG)),
        ApplyAggOp(Sum())(Ref(Name("x"), TInt64)),
      ),
      AggFilter(scan ceq I64(0), agg, false),
      AggGroupBy(scan, agg, false),
      AggExplode(ToStream(MakeArray(ArraySeq(scan), TArray(TInt64))), Name("e"), agg, false),
      AggArrayPerElement(
        MakeArray(ArraySeq(scan), TArray(TInt64)),
        Name("e"),
        Name("i"),
        agg,
        None,
        false,
      ),
      aggFoldIR(I64(0))(acc => acc + scan)(_ + _),
    )
  }

  @ParameterizedTest("scansInsideAggs")
  def testScanInsideAggFails(badIr: IR)(implicit ctx: ExecuteContext): Unit =
    interceptException[AssertionError]("ApplyScanOp outside a scan scope") {
      TypeCheck(ctx, badIr, BindingEnv.empty.createAgg.createScan)
    }
}
