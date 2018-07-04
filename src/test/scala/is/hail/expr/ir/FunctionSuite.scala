package is.hail.expr.ir

import java.io.PrintWriter

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions}
import is.hail.expr.types._
import is.hail.TestUtils._
import org.testng.annotations.Test
import is.hail.expr.{EvalContext, Parser}
import is.hail.table.Table
import is.hail.utils.FastSeq
import is.hail.variant.Call2

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
    registerIR("addone", TInt32())(ApplyBinaryPrimOp(Add(), _, I32(1)))
    registerIR("sumaggregator32", TAggregable(TInt32())) { ir =>
      val aggSig = AggSignature(Sum(), FastSeq(), None, FastSeq(TInt64()))
      ApplyAggOp(SeqOp(I32(0), FastSeq(Cast(ir, TInt64())), aggSig), FastSeq(), None, aggSig)
    }
    registerJavaStaticFunction("compare", TInt32(), TInt32(), TInt32())(classOf[java.lang.Integer], "compare")
    registerScalaFunction("foobar1", TInt32())(ScalaTestObject.getClass, "testFunction")
    registerScalaFunction("foobar2", TInt32())(ScalaTestCompanion.getClass, "testFunction")
    registerCode("testCodeUnification", tnum("x"), tv("x", _.isInstanceOf[TInt32]), tv("x")){ (_, a: Code[Int], b: Code[Int]) => a + b }
    registerCode("testCodeUnification2", tv("x"), tv("x")){ case (_, a: Code[Long]) => a }
  }
}

class FunctionSuite extends SparkSuite {

  val ec = EvalContext()
  val region = Region()

  TestRegisterFunctions.registerAll()

  def emitFromFB[F >: Null : TypeInfo](fb: FunctionBuilder[F]) =
    new EmitFunctionBuilder[F](fb.parameterTypeInfo, fb.returnTypeInfo, fb.packageName)

  def fromHailString(hql: String): IR =
    Parser.parseToAST(hql, ec).toIROptNoWarning().get

  def toF[R: TypeInfo](ir: IR): AsmFunction1[Region, R] = {
    val fb = emitFromFB(FunctionBuilder.functionBuilder[Region, R])
    Emit(ir, fb)
    fb.result(Some(new PrintWriter(System.out)))()
  }

  def toF[A: TypeInfo, R: TypeInfo](ir: IR): AsmFunction3[Region, A, Boolean, R] = {
    val fb = emitFromFB(FunctionBuilder.functionBuilder[Region, A, Boolean, R])
    Emit(ir, fb)
    fb.result(Some(new PrintWriter(System.out)))()
  }

  def lookup(meth: String, types: Type*)(irs: IR*): IR =
    IRFunctionRegistry.lookupConversion(meth, types).get(irs)

  @Test
  def testCodeFunction() {
    val ir = MakeStruct(Seq(("x", lookup("triangle", TInt32())(In(0, TInt32())))))
    val f = toF[Int, Long](ir)
    val off = f(region, 5, false)
    val expected = (5 * (5 + 1)) / 2
    val actual = region.loadInt(TStruct("x"-> TInt32()).loadField(region, off, 0))
    assert(actual == expected)
  }

  @Test
  def testStaticFunction() {
    val ir = lookup("compare", TInt32(), TInt32())(In(0, TInt32()), I32(0))
    val f = toF[Int, Int](ir)
    val actual = f(region, 5, false)
    assert(actual > 0)
  }

  @Test
  def testScalaFunction() {
    val ir = lookup("foobar1")()
    val f = toF[Int](ir)
    val actual = f(region)
    assert(actual == 1)
  }

  @Test
  def testIRConversion() {
    val ir = lookup("addone", TInt32())(In(0, TInt32()))
    val f = toF[Int, Int](ir)
    val actual = f(region, 5, false)
    assert(actual == 6)
  }

  @Test
  def testAggregatorConversion() {
    val t = Table.range(hc, 10)

    val tagg = TAggregable(t.signature)
    val idxField = GetField(Ref("row", t.signature), "idx")
    val ir = lookup("sumaggregator32", TAggregable(TInt32()))(idxField)

    val actual = Interpret[Long](TableAggregate(t.tir, ir))
    assert(actual == 45)
  }

  @Test
  def testScalaFunctionCompanion() {
    val ir = lookup("foobar2")()
    val f = toF[Int](ir)
    val actual = f(region)
    assert(actual == 2)
  }

  @Test
  def testUnifySize() {
    val ir = lookup("size", TArray(TInt32()))(In(0, TArray(TInt32())))
    val f = toF[Long, Int](ir)
    val rvb = new RegionValueBuilder(region)
    rvb.start(TArray(TInt32()))
    rvb.addAnnotation(TArray(TInt32()), IndexedSeq(0, 1, 2, 3))
    val actual = f(region, rvb.end(), false)
    assert(actual == 4)
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
    val ir = lookup("UnphasedDiploidGtIndexCall", TInt32())(In(0, TInt32()))
    val f = toF[Int, Int](ir)
    assert(f(region, 0, false) == Call2.fromUnphasedDiploidGtIndex(0))
  }
}
