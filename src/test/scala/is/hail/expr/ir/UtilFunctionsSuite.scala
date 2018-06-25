package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.types._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class UtilFunctionsSuite extends TestNGSuite {
  val na = NA(TBoolean())
  val die = Die("it ded", TBoolean())
  val folded = ArrayFold(
    MakeArray(Seq(true), TArray(TBoolean())),
    die, "a", "e",
    Ref("a", TBoolean()) || Ref("e", TBoolean()))

  @Test def shortCircuitOr() {
    assertEvalsTo(True() || True(), true)
    assertEvalsTo(True() || False(), true)
    assertEvalsTo(False() || True(), true)
    assertEvalsTo(False() || False(), false)
  }

  @Test def shortCircuitOrHandlesMissingness() {
    assertEvalsTo(na || na, null)
    assertEvalsTo(na || True(), true)
    assertEvalsTo(True() || na, true)
    assertEvalsTo(na || False(), null)
    assertEvalsTo(False() || na, null)

  }

  @Test def shortCircuitOrHandlesErrors() {
    //FIXME: interpreter evaluates args for ApplySpecial before invoking the function :-|
    assertCompiledFatal(na || die, "it ded")
    assertCompiledFatal(False() || die, "it ded")
    assert(eval(True() || die) == true)

    assertCompiledFatal(die || na, "it ded")
    assertCompiledFatal(die || False(), "it ded")
    assertCompiledFatal(die || True(), "it ded")

    // ArrayFold does computation in setup; this ensures r.setup is not being called if unnecessary.
    assertCompiledFatal(na || folded, "it ded")
    assertCompiledFatal(False() || folded, "it ded")
    assert(eval(True() || folded) == true)
  }

  @Test def shortCircuitAnd() {
    assertEvalsTo(True() && True(), true)
    assertEvalsTo(True() && False(), false)
    assertEvalsTo(False() && True(), false)
    assertEvalsTo(False() && False(), false)
  }

  @Test def shortCircuitAndHandlesMissingness() {
    assertEvalsTo(na && na, null)
    assertEvalsTo(True() && na, null)
    assertEvalsTo(na && True(), null)
    assertEvalsTo(False() && na, false)
    assertEvalsTo(na && False(), false)
  }

  @Test def shortCircuitAndHandlesErroes() {
    //FIXME: interpreter evaluates args for ApplySpecial before invoking the function :-|
    assertCompiledFatal(na && die, "it ded")
    assertCompiledFatal(True() && die, "it ded")
    assert(eval(False() && die) == false)

    assertCompiledFatal(die && na, "it ded")
    assertCompiledFatal(die && True(), "it ded")
    assertCompiledFatal(die && False(), "it ded")

    // ArrayFold does computation in setup; this ensures r.setup is not being called if unnecessary.
    assertCompiledFatal(na && folded, "it ded")
    assertCompiledFatal(True() && folded, "it ded")
    assert(eval(False() && folded) == false)
  }
}
