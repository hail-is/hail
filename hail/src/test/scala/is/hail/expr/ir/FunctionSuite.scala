package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.{ExecStrategy, HailSuite}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions}
import is.hail.expr.types.virtual._
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
    registerIR("addone", TInt32, TInt32)(ApplyBinaryPrimOp(Add(), _, I32(1)))
    registerJavaStaticFunction("compare", Array(TInt32, TInt32), TInt32, null)(classOf[java.lang.Integer], "compare")
    registerScalaFunction("foobar1", Array(), TInt32, null)(ScalaTestObject.getClass, "testFunction")
    registerScalaFunction("foobar2", Array(), TInt32, null)(ScalaTestCompanion.getClass, "testFunction")
    registerCode[Int, Int]("testCodeUnification", tnum("x"), tv("x", "int32"), tv("x"), null) {
      case (_, rt, (aT, a: Code[Int]), (bT, b: Code[Int])) => a + b
    }
    registerCode("testCodeUnification2", tv("x"), tv("x"), null) { case (_, rt, (aT, a: Code[Long])) => a }
  }
}

class FunctionSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly
  val region = Region()

  TestRegisterFunctions.registerAll()

  def lookup(meth: String, rt: Type, types: Type*)(irs: IR*): IR =
    IRFunctionRegistry.lookupConversion(meth, rt, types).get(irs)

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
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", TInt32, Seq(TInt32, TInt32)).isDefined)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", TInt32, Seq(TInt64, TInt32)).isEmpty)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", TInt64, Seq(TInt32, TInt32)).isEmpty)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification2", TArray(TInt32), Seq(TArray(TInt32))).isDefined)
  }

  @Test
  def testUnphasedDiploidGtIndexCall() {
    assertEvalsTo(lookup("UnphasedDiploidGtIndexCall", TCall, TInt32)(In(0, TInt32)),
      FastIndexedSeq(0 -> TInt32),
      Call2.fromUnphasedDiploidGtIndex(0))
  }

  @Test
  def testFunctionBuilderGetOrDefine() {
    val fb = EmitFunctionBuilder[Int]("foo")
    val i = fb.genFieldThisRef[Int]()
    val mb1 = fb.getOrGenMethod("foo", "foo", Array[TypeInfo[_]](), UnitInfo) { mb =>
      mb.emit(i := i + 1)
    }
    val mb2 = fb.getOrGenMethod("foo", "foo", Array[TypeInfo[_]](), UnitInfo) { mb =>
      mb.emit(i := i - 100)
    }
    fb.emit(Code(i := 0, mb1.invoke(), mb2.invoke(), i))
    Region.scoped { r =>

      assert(fb.resultWithIndex().apply(0, r)() == 2)
    }
  }

  @Test def testFunctionBuilderWrapVoids() {
    val fb = EmitFunctionBuilder[Int]("foo")
    val i = fb.genFieldThisRef[Int]()

    val codes = Array(
      i := i + 1,
      i := i + 2,
      i := i + 3,
      i := i + 4,
      i := i + 5,
      i := i + 6
    )

    fb.emit(Code(i := 0, fb.wrapVoids(codes, "foo", 2), i))
    Region.smallScoped { r =>
      assert(fb.resultWithIndex().apply(0, r).apply() == 21)

    }
  }

  @Test def testFunctionBuilderWrapVoidsWithArgs() {
    val fb = EmitFunctionBuilder[Int]("foo")
    val i = fb.newLocal[Int]()
    val j = fb.genFieldThisRef[Int]()

    val codes = Array[Seq[Code[_]] => Code[Unit]](
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(1) * ii },
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(2) * ii },
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(3) * ii },
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(4) * ii },
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(5) * ii },
      { case Seq(ii: Code[Int@unchecked]) => j := j + const(6) * ii }
    )

    fb.emit(Code(j := 0, i := 1, fb.wrapVoidsWithArgs(codes, "foo", Array(IntInfo), Array(i.load()), 2), j))
    Region.smallScoped { r =>
      assert(fb.resultWithIndex().apply(0, r).apply() == 21)
    }
  }
}
