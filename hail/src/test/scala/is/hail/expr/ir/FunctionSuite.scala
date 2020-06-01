package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.{ExecStrategy, HailSuite}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions}
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.Call2
import org.testng.annotations.Test
import is.hail.TestUtils._

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
  def registerAll() {
    registerIR1("addone", TInt32, TInt32)((_, a) => ApplyBinaryPrimOp(Add(), a, I32(1)))
    registerJavaStaticFunction("compare", Array(TInt32, TInt32), TInt32, null)(classOf[java.lang.Integer], "compare")
    registerScalaFunction("foobar1", Array(), TInt32, null)(ScalaTestObject.getClass, "testFunction")
    registerScalaFunction("foobar2", Array(), TInt32, null)(ScalaTestCompanion.getClass, "testFunction")
    registerCode2[Int, Int]("testCodeUnification", tnum("x"), tv("x", "int32"), tv("x"), null) {
      case (_, rt, (aT, a: Code[Int]), (bT, b: Code[Int])) => a + b
    }
    registerCode1("testCodeUnification2", tv("x"), tv("x"), null) { case (_, rt, (aT, a: Code[Long])) => a }
  }
}

class FunctionSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly
  val region = Region()

  TestRegisterFunctions.registerAll()

  def lookup(meth: String, rt: Type, types: Type*)(irs: IR*): IR = {
    val l = IRFunctionRegistry.lookupUnseeded(meth, rt, types).get
    l(Seq(), irs)
  }

  @Test
  def testCodeFunction() {
    assertEvalsTo(lookup("triangle", TInt32, TInt32)(In(0, TInt32)),
      FastIndexedSeq(5 -> TInt32),
      (5 * (5 + 1)) / 2)
  }

  @Test
  def testStaticFunction() {
    assertEvalsTo(lookup("compare", TInt32, TInt32, TInt32)(In(0, TInt32), I32(0)) > 0,
      FastIndexedSeq(5 -> TInt32),
      true)
  }

  @Test
  def testScalaFunction() {
    assertEvalsTo(lookup("foobar1", TInt32)(), 1)
  }

  @Test
  def testIRConversion() {
    assertEvalsTo(lookup("addone", TInt32, TInt32)(In(0, TInt32)),
      FastIndexedSeq(5 -> TInt32),
      6)
  }

  @Test
  def testScalaFunctionCompanion() {
    assertEvalsTo(lookup("foobar2", TInt32)(), 2)
  }

  @Test
  def testVariableUnification() {
    assert(IRFunctionRegistry.lookupUnseeded("testCodeUnification", TInt32, Seq(TInt32, TInt32)).isDefined)
    assert(IRFunctionRegistry.lookupUnseeded("testCodeUnification", TInt32, Seq(TInt64, TInt32)).isEmpty)
    assert(IRFunctionRegistry.lookupUnseeded("testCodeUnification", TInt64, Seq(TInt32, TInt32)).isEmpty)
    assert(IRFunctionRegistry.lookupUnseeded("testCodeUnification2", TArray(TInt32), Seq(TArray(TInt32))).isDefined)
  }

  @Test
  def testUnphasedDiploidGtIndexCall() {
    assertEvalsTo(lookup("UnphasedDiploidGtIndexCall", TCall, TInt32)(In(0, TInt32)),
      FastIndexedSeq(0 -> TInt32),
      Call2.fromUnphasedDiploidGtIndex(0))
  }

  @Test
  def testFunctionBuilderGetOrDefine() {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val i = fb.genFieldThisRef[Int]()
    val mb1 = fb.getOrGenEmitMethod("foo", "foo", FastIndexedSeq[ParamType](), UnitInfo) { mb =>
      mb.emit(i := i + 1)
    }
    val mb2 = fb.getOrGenEmitMethod("foo", "foo", FastIndexedSeq[ParamType](), UnitInfo) { mb =>
      mb.emit(i := i - 100)
    }
    fb.emit(Code(i := 0, mb1.invokeCode(), mb2.invokeCode(), i))
    Region.scoped { r =>

      assert(fb.resultWithIndex().apply(0, r)() == 2)
    }
  }

  @Test def testFunctionBuilderWrapVoids() {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val i = fb.genFieldThisRef[Int]()

    val codes = Array(
      i := i + 1,
      i := i + 2,
      i := i + 3,
      i := i + 4,
      i := i + 5,
      i := i + 6
    )

    fb.emitWithBuilder { cb =>
      cb.assign(i, 0)
      fb.wrapVoids(cb, codes.map(x => (cb: EmitCodeBuilder) => cb += x), "foo", 2)
      i
    }
    Region.smallScoped { r =>
      assert(fb.resultWithIndex().apply(0, r).apply() == 21)

    }
  }

  @Test def testFunctionBuilderWrapVoidsWithArgs() {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val i = fb.newLocal[Int]()
    val j = fb.genFieldThisRef[Int]()

    val codes = Array[(EmitCodeBuilder, Seq[Code[_]]) => Unit](
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(1) * ii) },
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(2) * ii) },
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(3) * ii) },
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(4) * ii) },
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(5) * ii) },
      { case (cb, Seq(ii: Code[Int@unchecked])) => cb += (j := j + const(6) * ii) }
    )

    fb.emitWithBuilder { cb =>
      cb.assign(j, 0)
      cb.assign(i, 1)
      fb.wrapVoidsWithArgs(cb, codes, "foo", Array(IntInfo), Array(i.load()), 2)
      j
    }
    Region.smallScoped { r =>
      assert(fb.resultWithIndex().apply(0, r).apply() == 21)
    }
  }
}
