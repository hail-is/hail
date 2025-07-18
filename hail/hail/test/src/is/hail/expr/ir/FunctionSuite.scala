package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.asm4s._
import is.hail.expr.ir.defs.{ApplyBinaryPrimOp, I32, In}
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils.FastSeq
import is.hail.variant.Call2

import org.scalatest
import org.testng.annotations.Test

object ScalaTestObject {
  def testFunction(): Int = 1
}

object ScalaTestCompanion {
  def testFunction(): Int = 2
}

class ScalaTestCompanion {
  def testFunction(): Int = 3
}

object TestRegisterFunctions extends RegistryFunctions {
  def registerAll(): Unit = {
    registerIR1("addone", TInt32, TInt32)((_, a, _) => ApplyBinaryPrimOp(Add(), a, I32(1)))
    registerJavaStaticFunction("compare", Array(TInt32, TInt32), TInt32, null)(
      classOf[java.lang.Integer],
      "compare",
    )
    registerScalaFunction("foobar1", Array(), TInt32, null)(
      ScalaTestObject.getClass,
      "testFunction",
    )
    registerScalaFunction("foobar2", Array(), TInt32, null)(
      ScalaTestCompanion.getClass,
      "testFunction",
    )
    registerSCode2("testCodeUnification", tnum("x"), tv("x", "int32"), tv("x"), null) {
      case (_, cb, _, a, b, _) => primitive(cb.memoize(a.asInt.value + b.asInt.value))
    }
    registerSCode1("testCodeUnification2", tv("x"), tv("x"), null) { case (_, _, _, a, _) => a }
  }
}

class FunctionSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  TestRegisterFunctions.registerAll()

  @Test
  def testCodeFunction(): scalatest.Assertion =
    assertEvalsTo(
      invoke("triangle", TInt32, In(0, TInt32)),
      FastSeq(5 -> TInt32),
      (5 * (5 + 1)) / 2,
    )

  @Test
  def testStaticFunction(): scalatest.Assertion =
    assertEvalsTo(
      invoke("compare", TInt32, In(0, TInt32), I32(0)) > 0,
      FastSeq(5 -> TInt32),
      true,
    )

  @Test
  def testScalaFunction(): scalatest.Assertion =
    assertEvalsTo(invoke("foobar1", TInt32), 1)

  @Test
  def testIRConversion(): scalatest.Assertion =
    assertEvalsTo(invoke("addone", TInt32, In(0, TInt32)), FastSeq(5 -> TInt32), 6)

  @Test
  def testScalaFunctionCompanion(): scalatest.Assertion =
    assertEvalsTo(invoke("foobar2", TInt32), 2)

  @Test
  def testVariableUnification(): scalatest.Assertion = {
    assert(IRFunctionRegistry.lookupUnseeded(
      "testCodeUnification",
      TInt32,
      Seq(TInt32, TInt32),
    ).isDefined)
    assert(IRFunctionRegistry.lookupUnseeded(
      "testCodeUnification",
      TInt32,
      Seq(TInt64, TInt32),
    ).isEmpty)
    assert(IRFunctionRegistry.lookupUnseeded(
      "testCodeUnification",
      TInt64,
      Seq(TInt32, TInt32),
    ).isEmpty)
    assert(IRFunctionRegistry.lookupUnseeded(
      "testCodeUnification2",
      TArray(TInt32),
      Seq(TArray(TInt32)),
    ).isDefined)
  }

  @Test
  def testUnphasedDiploidGtIndexCall(): scalatest.Assertion =
    assertEvalsTo(
      invoke("UnphasedDiploidGtIndexCall", TCall, In(0, TInt32)),
      FastSeq(0 -> TInt32),
      Call2.fromUnphasedDiploidGtIndex(0),
    )

  @Test
  def testGetOrGenMethod(): scalatest.Assertion = {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val i = fb.genFieldThisRef[Int]()
    val mb1 = fb.getOrGenEmitMethod("foo", "foo", FastSeq[ParamType](), UnitInfo) { mb =>
      mb.emit(i := i + 1)
    }
    val mb2 = fb.getOrGenEmitMethod("foo", "foo", FastSeq[ParamType](), UnitInfo) { mb =>
      mb.emit(i := i - 100)
    }
    fb.emitWithBuilder { cb =>
      cb.assign(i, 0)
      cb.invokeVoid(mb1, cb.this_)
      cb.invokeVoid(mb2, cb.this_)
      i
    }
    pool.scopedRegion { r =>
      assert(fb.resultWithIndex().apply(theHailClassLoader, ctx.fs, ctx.taskContext, r)() == 2)
    }
  }
}
