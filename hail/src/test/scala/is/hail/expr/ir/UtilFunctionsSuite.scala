package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.types.virtual.{TBoolean, TInt32, TStream}

import org.testng.annotations.Test

class UtilFunctionsSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  val na = NA(TBoolean)
  val die = Die("it ded", TBoolean)

  val folded = StreamFold(
    MakeStream(IndexedSeq(true), TStream(TBoolean)),
    die,
    "a",
    "e",
    Ref("a", TBoolean) || Ref("e", TBoolean),
  )

  @Test def shortCircuitOr(): Unit = {
    assertEvalsTo(True() || True(), true)
    assertEvalsTo(True() || False(), true)
    assertEvalsTo(False() || True(), true)
    assertEvalsTo(False() || False(), false)
  }

  @Test def shortCircuitOrHandlesMissingness(): Unit = {
    assertEvalsTo(na || na, null)
    assertEvalsTo(na || True(), true)
    assertEvalsTo(True() || na, true)
    assertEvalsTo(na || False(), null)
    assertEvalsTo(False() || na, null)

  }

  @Test def shortCircuitOrHandlesErrors(): Unit = {
    // FIXME: interpreter evaluates args for ApplySpecial before invoking the function :-|
    assertCompiledFatal(na || die, "it ded")
    assertCompiledFatal(False() || die, "it ded")
    // FIXME: This needs to be fixed with an interpreter function registry
    // assert(eval(True() || die) == true)

    assertCompiledFatal(die || na, "it ded")
    assertCompiledFatal(die || False(), "it ded")
    assertCompiledFatal(die || True(), "it ded")

    // ArrayFold does computation in setup; this ensures r.setup is not being called if unnecessary.
    assertCompiledFatal(na || folded, "it ded")
    assertCompiledFatal(False() || folded, "it ded")
    assert(eval(True() || folded) == true)
  }

  @Test def shortCircuitAnd(): Unit = {
    assertEvalsTo(True() && True(), true)
    assertEvalsTo(True() && False(), false)
    assertEvalsTo(False() && True(), false)
    assertEvalsTo(False() && False(), false)
  }

  @Test def shortCircuitAndHandlesMissingness(): Unit = {
    assertEvalsTo(na && na, null)
    assertEvalsTo(True() && na, null)
    assertEvalsTo(na && True(), null)
    assertEvalsTo(False() && na, false)
    assertEvalsTo(na && False(), false)
  }

  @Test def shortCircuitAndHandlesErroes(): Unit = {
    // FIXME: interpreter evaluates args for ApplySpecial before invoking the function :-|
    assertCompiledFatal(na && die, "it ded")
    assertCompiledFatal(True() && die, "it ded")
    // assert(eval(False() && die) == false)

    assertCompiledFatal(die && na, "it ded")
    assertCompiledFatal(die && True(), "it ded")
    assertCompiledFatal(die && False(), "it ded")

    // ArrayFold does computation in setup; this ensures r.setup is not being called if unnecessary.
    assertCompiledFatal(na && folded, "it ded")
    assertCompiledFatal(True() && folded, "it ded")
    assert(eval(False() && folded) == false)
  }

  @Test def testParseFunctionRequiredness(): Unit = {
    assertEvalsTo(invoke("toInt32OrMissing", TInt32, Str("123")), 123)
    assertEvalsTo(invoke("toInt32OrMissing", TInt32, Str("foo")), null)
  }
}
