package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions, UtilFunctions}
import is.hail.expr.types._
import org.testng.annotations.Test
import is.hail.expr.{EvalContext, Parser}

import scala.reflect.ClassTag

object ScalaTestObject {
  def testFunction(): Int = 1
}

object ScalaTestCompanion {
  def testFunction(): Int = 1
}

class ScalaTestCompanion {
  def testFunction(): Int = 1
}


object TestRegisterFunctions extends RegistryFunctions {

  registerJavaStaticFunction[java.lang.Integer, Int]("compare", "compare", TInt32(), TInt32(), TInt32())
  registerScalaFunction[Int]("foobar1", ScalaTestObject, "testFunction", TInt32())
  registerScalaFunction[Int]("foobar2", ScalaTestCompanion, "testFunction", TInt32())


}

class FunctionSuite {

  TestRegisterFunctions.registerAll()

  val ec = EvalContext()
  val region = Region()

  def fromHailString(hql: String): IR = Parser.parseToAST(hql, ec).toIR().get

  def toF[R: TypeInfo](ir: IR): AsmFunction1[Region, R] = {
    Infer(ir)
    val fb = FunctionBuilder.functionBuilder[Region, R]
    Emit(ir, fb)
    fb.result(Some(new PrintWriter(System.out)))()
  }

  def toF[A: TypeInfo, R: TypeInfo](ir: IR): AsmFunction3[Region, A, Boolean, R] = {
    Infer(ir)
    val fb = FunctionBuilder.functionBuilder[Region, A, Boolean, R]
    Emit(ir, fb)
    fb.result(Some(new PrintWriter(System.out)))()
  }

  @Test
  def testCodeFunction() {
    UtilFunctions.registerAll()
    val ir = MakeStruct(Seq(("x", IRFunctionRegistry.lookupFunction("triangle", Seq(TInt32(), TInt32())).get(Seq(In(0, TInt32()))))))
    val f = toF[Int, Long](ir)
    val off = f(region, 5, false)
    val expected = (5 * (5 + 1)) / 2
    val actual = region.loadInt(TStruct("x"-> TInt32()).loadField(region, off, 0))
    assert(actual == expected)
  }

  @Test
  def testStaticFunction() {
    val ir = IRFunctionRegistry.lookupFunction("compare", Seq(TInt32(), TInt32(), TInt32())).get(Seq(In(0, TInt32()), I32(0)))
    val f = toF[Int, Int](ir)
    val actual = f(region, 5, false)
    assert(actual > 0)
  }

  @Test
  def testScalaFunction() {
    val ir = IRFunctionRegistry.lookupFunction("foobar1", Seq(TInt32())).get(Seq())
    val f = toF[Int](ir)
    val actual = f(region)
    assert(actual == 1)
  }

  @Test
  def testScalaFunctionCompanion() {
    val ir = IRFunctionRegistry.lookupFunction("foobar2", Seq(TInt32())).get(Seq())
    val f = toF[Int](ir)
    val actual = f(region)
    assert(actual == 1)
  }

}
