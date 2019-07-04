package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.{ExecStrategy, HailSuite}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions}
import is.hail.expr.types.virtual._
import is.hail.utils.FastIndexedSeq
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
    registerIR("addone", TInt32(), TInt32())(ApplyBinaryPrimOp(Add(), _, I32(1)))
    registerJavaStaticFunction("compare", TInt32(), TInt32(), TInt32())(classOf[java.lang.Integer], "compare")
    registerScalaFunction("foobar1", TInt32())(ScalaTestObject.getClass, "testFunction")
    registerScalaFunction("foobar2", TInt32())(ScalaTestCompanion.getClass, "testFunction")
    registerCode[Int, Int]("testCodeUnification", tnum("x"), tv("x", "int32"), tv("x")){
      case (_, (aT, a: Code[Int]), (bT, b: Code[Int]))  => a + b }
    registerCode("testCodeUnification2", tv("x"), tv("x")){ case (_, (aT, a: Code[Long])) => a }
  }
}

class FunctionSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly
  val region = Region()

  TestRegisterFunctions.registerAll()

  def emitFromFB[F >: Null : TypeInfo](fb: FunctionBuilder[F]) =
    new EmitFunctionBuilder[F](fb.parameterTypeInfo, fb.returnTypeInfo, fb.packageName)

  def lookup(meth: String, types: Type*)(irs: IR*): IR =
    IRFunctionRegistry.lookupConversion(meth, types).get(irs)

  @Test
  def testCodeFunction() {
    assertEvalsTo(lookup("triangle", TInt32())(In(0, TInt32())),
      FastIndexedSeq(5 -> TInt32()),
      (5 * (5 + 1)) / 2)
  }

  @Test
  def testStaticFunction() {
    assertEvalsTo(lookup("compare", TInt32(), TInt32())(In(0, TInt32()), I32(0)) > 0,
      FastIndexedSeq(5 -> TInt32()),
      true)
  }

  @Test
  def testScalaFunction() {
    assertEvalsTo(lookup("foobar1")(), 1)
  }

  @Test
  def testIRConversion() {
    assertEvalsTo(lookup("addone", TInt32())(In(0, TInt32())),
      FastIndexedSeq(5 -> TInt32()),
      6)
  }

  @Test
  def testScalaFunctionCompanion() {
    assertEvalsTo(lookup("foobar2")(), 2)
  }

  @Test
  def testVariableUnification() {
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", Seq(TInt32(), TInt32())).isDefined)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", Seq(TInt64(), TInt32())).isEmpty)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification", Seq(TInt64(), TInt64())).isEmpty)
    assert(IRFunctionRegistry.lookupConversion("testCodeUnification2", Seq(TArray(TInt32()))).isDefined)
  }

  @Test
  def testUnphasedDiploidGtIndexCall() {
    assertEvalsTo(lookup("UnphasedDiploidGtIndexCall", TInt32())(In(0, TInt32())),
      FastIndexedSeq(0 -> TInt32()),
      Call2.fromUnphasedDiploidGtIndex(0))
  }
}
