package is.hail.methods.ir

import is.hail.utils.using
import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.ir._
import is.hail.expr.ir.PointedTypeInfo._
import is.hail.expr.ir.IR._
import is.hail.expr.ir.Compile.DetailedTypeInfo
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

import scala.reflect.ClassTag

import java.io.PrintWriter

class CompileSuite {
  def compileAndRun[T: TypeInfo](mgti: DetailedTypeInfo[T, _], ir: IR, write: Boolean = false): T = {
    val fb = FunctionBuilder.functionBuilder[T]
    Compile(ir, Array(mgti)).emit(fb)
    fb.result().apply().apply()
  }

  private def unboxed[T: TypeInfo] = DetailedTypeInfo[T, T](None)

  private def boxed[T : ClassTag : TypeInfo, UT : ClassTag : TypeInfo] = DetailedTypeInfo[T, UT](Some(
    (typeInfo[T], x => ucode.Erase(Code.newInstance[T, UT](ucode.Reify[UT](x))))))

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
    assert(compileAndRun(unboxed[Int], Out1(Let("x", I32(10), IntInfo, Ref("x")))) === 10)
  }

  @Test(enabled = false)
  def arrays() {
    assert(compileAndRun(unboxed[Array[Int]], Out1(MakeArray(Array(), IntInfo))) === Array[Int]())
    assert(compileAndRun(unboxed[Array[Int]], Out1(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo))) === Array(4,3,2,1,0))
    assert(compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(0), IntInfo))) === 4)
    assert(compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(4), IntInfo))) === 0)
    intercept[IndexOutOfBoundsException] {
      compileAndRun(unboxed[Int], Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(5), IntInfo)))
    }
  }

  @Test
  def simpleNullnessInt() {
    assert(compileAndRun(boxed[Integer, Int], Out1(_NA[Int])) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(_NA[Int])) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(True(), _NA[Int], _NA[Int]))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), _NA[Int], _NA[Int]))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(Let("x", _NA[Int], typeInfo[Int], Ref("x")))) === null)
    assert(compileAndRun(unboxed[Int], Out1(Let("x", _NA[Int], typeInfo[Int], I32(1)))) === 1)
    assert(compileAndRun(boxed[Integer, Int], Out1(Let("x", I32(2), IntInfo, _NA[Int]))) === null)
  }

  @Test
  def simpleNullnessOtherTypes() {
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(_NA[Double])) === null)
    assert(compileAndRun(boxed[java.lang.Float, Float], Out1(_NA[Float])) === null)
    assert(compileAndRun(boxed[java.lang.Long, Long], Out1(If(True(), _NA[Long], _NA[Long]))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), _NA[Int], _NA[Int]))) === null)
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(Let("x", _NA[Double], typeInfo[Double], Ref("x")))) === null)
    assert(compileAndRun(unboxed[Float], Out1(Let("x", _NA[Int], typeInfo[Int], F32(1.0f)))) === 1.0)
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(Let("x", F64(2.0), DoubleInfo, _NA[Double]))) === null)
  }

  @Test
  def mapNAInt() {
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", _NA[Int], typeInfo[Int], I32(1), pointedTypeInfo[Int]))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", _NA[Int], typeInfo[Int], Ref("x"), pointedTypeInfo[Int]))) === null)
    assert(compileAndRun(boxed[Integer, Int], Out1(MapNA("x", I32(3), typeInfo[Int], Ref("x"), pointedTypeInfo[Int]))) === 3)
  }

  @Test
  def mapNAOtherTypes() {
    assert(compileAndRun(boxed[java.lang.Double, Double], Out1(MapNA("x", _NA[Double], typeInfo[Double], F64(1.0), pointedTypeInfo[Double]))) === null)
    assert(compileAndRun(boxed[java.lang.Long, Long], Out1(MapNA("x", _NA[Long], typeInfo[Long], Ref("x"), pointedTypeInfo[Long]))) === null)
    assert(compileAndRun(boxed[java.lang.Float, Float], Out1(MapNA("x", F32(3.0f), typeInfo[Float], Ref("x"), pointedTypeInfo[Float]))) === 3)
  }

  @Test
  def differingNullnessOnBranchesOfAnIf() {
    assert(compileAndRun(boxed[Integer, Int], Out1(If(False(), _NA[Int], I32(3))), true) === 3)
  }

}
