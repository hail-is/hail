package is.hail.scalacheck

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BindingEnv, TypeCheck}

import org.junit.jupiter.api.Test
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class ArbitraryIRSuite {

  private val genValueIR =
    for {
      t <- ArbitraryIR.genType()
      ir <- Gen.sized(ArbitraryIR.genIR(t, BindingEnv.empty, _))
    } yield ir

  @Test def testValueIRsTypeCheck(implicit ctx: ExecuteContext): Unit =
    check(forAll(genValueIR)(TypeCheck(ctx, _)))

  @Test def testTableIRsTypeCheck(implicit ctx: ExecuteContext): Unit =
    check(forAll(Gen.sized(ArbitraryIR.genTable))(TypeCheck(ctx, _)))

  @Test def testMatrixIRsTypeCheck(implicit ctx: ExecuteContext): Unit =
    check(forAll(Gen.sized(ArbitraryIR.genMatrix))(TypeCheck(ctx, _)))
}
