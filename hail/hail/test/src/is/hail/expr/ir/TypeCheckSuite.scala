package is.hail.expr.ir

import is.hail.ParameterizedTest
import is.hail.TestUtils.interceptException
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs._

import scala.collection.immutable.ArraySeq

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
}
