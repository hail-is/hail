package is.hail.expr.ir

import is.hail.expr.types.virtual.{TInt32, TString, TVariable}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class TVariableSuite extends TestNGSuite {

  @Test def testMutability(): Unit = {
    val tv1 = TVariable("T")
    val tv2 = TVariable("T")

    tv1.unify(TInt32())
    val t1 = tv1.subst()
    assert(t1 == TInt32())

    tv2.clear()
    tv2.unify(TString())
    val t2 = tv2.subst()
    assert(t2 == TString())

    assert(t1 == tv1.subst())
    assert(t2 == tv2.subst())
  }
}
