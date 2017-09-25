package is.hail.methods.ir

import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.{TInt32, TInt64, TFloat32, TFloat64}
import is.hail.expr.ir._
import is.hail.expr.ir.IR._
import is.hail.expr.ir.Compile.DetailedTypeInfo
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

import scala.reflect.ClassTag

import java.io.PrintWriter

class CompileSuite {
  def compileAndRun[T: TypeInfo](mgti: DetailedTypeInfo[T, _], ir: IR, print: Option[PrintWriter] = None): T = {
    val fb = FunctionBuilder.functionBuilder[T]
    Compile(ir, Array(mgti), fb)
    fb.result(print).apply().apply()
  }

  private def unboxed[T: TypeInfo] = DetailedTypeInfo[T, T](None)

  private def boxed[T : ClassTag : TypeInfo, UT : ClassTag : TypeInfo] = DetailedTypeInfo[T, UT](Some(
    (typeInfo[T], x => Code.newInstance[T, UT](x.asInstanceOf[Code[UT]]))))

  @Test
  def constants() {
    assert(compileAndRun(unboxed[Int], Out1(I32(16))) === 16)
    assert(compileAndRun(unboxed[Long], Out1(I64(Int.MaxValue+1))) === Int.MaxValue+1)
    assert(compileAndRun(unboxed[Float], Out1(F32(0f))) === 0f)
    assert(compileAndRun(unboxed[Double], Out1(F64(Float.MaxValue * 2.0))) === Float.MaxValue * 2.0)
  }

  @Test
  def _if() {
    assert(compileAndRun(unboxed[Int], Out1(If(True(), I32(10), I32(-10)))) === 10)
    assert(compileAndRun(unboxed[Int], Out1(If(False(), I32(10), I32(-10)))) === -10)
  }

  @Test
  def nestedIf() {
    assert(compileAndRun(unboxed[Int], Out1(If(True(), If(True(), I32(3), I32(2)), I32(0)))) === 3)
    assert(compileAndRun(unboxed[Int], Out1(If(True(), If(False(), I32(3), I32(2)), I32(0)))) === 2)
    assert(compileAndRun(unboxed[Int], Out1(If(False(), If(True(), I32(3), I32(2)), I32(0)))) === 0)
  }

  @Test
  def let() {
    assert(compileAndRun(unboxed[Int], Out1(Let("x", I32(10), TInt32, Ref("x")))) === 10)
  }

  @Test(enabled = false)
  def arrays() {
    assert(compileAndRun(unboxed[Array[Int]], Out1(MakeArray(Array(), TInt32))) === Array[Int]())
    assert(compileAndRun(unboxed[Array[Int]], Out1(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), TInt32))) === Array(4,3,2,1,0))
    assert(compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), TInt32), I32(0), TInt32))) === 4)
    assert(compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), TInt32), I32(4), TInt32))) === 0)
    intercept[IndexOutOfBoundsException] {
      compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), TInt32), I32(5), TInt32)))
    }
  }

  @Test
  def simpleNullnessInt() {
    assert(compileAndRun(boxed[Integer, Int], Out1(NA(TInt32))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(NA(TInt32))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(True(), NA(TInt32), NA(TInt32)))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), NA(TInt32), NA(TInt32)))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(Let("x", NA(TInt32), TInt32, Ref("x")))) === null)
    assert(compileAndRun(unboxed[Int], Out1(Let("x", NA(TInt32), TInt32, I32(1)))) === 1)
    assert(compileAndRun(boxed[Integer, Int], Out1(Let("x", I32(2), TInt32, NA(TInt32)))) === null)
  }

  @Test
  def simpleNullnessOtherTypes() {
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(NA(TFloat64))) === null)
    assert(compileAndRun(boxed[java.lang.Float, Float], Out1(NA(TFloat32))) === null)
    assert(compileAndRun(boxed[java.lang.Long, Long], Out1(If(True(), NA(TInt64), NA(TInt64)))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), NA(TInt32), NA(TInt32)))) === null)
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(Let("x", NA(TFloat64), TFloat64, Ref("x")))) === null)
    assert(compileAndRun(unboxed[Float], Out1(Let("x", NA(TInt32), TInt32, F32(1.0f)))) === 1.0)
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(Let("x", F64(2.0), TFloat64, NA(TFloat64)))) === null)
  }

  @Test
  def mapNAInt() {
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", NA(TInt32), TInt32, I32(1), TInt32))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", NA(TInt32), TInt32, Ref("x"), TInt32))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", I32(3), TInt32, Ref("x"), TInt32))) === 3)
  }

  @Test
  def mapNAOtherTypes() {
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(MapNA("x", NA(TFloat64), TFloat64, F64(1.0), TFloat64))) === null)
    assert(compileAndRun(boxed[java.lang.Long, Long], Out1(MapNA("x", NA(TInt64), TInt64, Ref("x"), TInt64))) === null)
    assert(compileAndRun(boxed[java.lang.Float, Float], Out1(MapNA("x", F32(3.0f), TFloat32, Ref("x"), TFloat32))) === 3)
  }

  @Test
  def differingNullnessOnBranchesOfAnIf() {
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), NA(TInt32), I32(3))), Some(new PrintWriter(System.err))) === 3)
  }

}
