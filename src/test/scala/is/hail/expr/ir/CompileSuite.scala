package is.hail.methods.ir

import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.ir._
import is.hail.expr.ir.IR._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class CompileSuite {

  def compileAndRun[T: TypeInfo](mgti: MaybeGenericTypeInfo[T], ir: IR): T = {
    val fb = FunctionBuilder.functionBuilder[T]
    Compile(ir, Array(mgti)).run(fb).apply()
  }

  private def unboxed[T: TypeInfo]: MaybeGenericTypeInfo[T] =
    NotGenericTypeInfo()

  private def boxed[T: TypeInfo]: MaybeGenericTypeInfo[T] =
    GenericTypeInfo()

  @Test
  def constants() {
    assert(compileAndRun(unboxed[Int], Out1(I32(16))) === 16)
    assert(compileAndRun(unboxed[Long], Out1(I64(Int.MaxValue+1))) === Int.MaxValue+1)
    assert(compileAndRun(unboxed[Float], Out1(F32(0f))) === 0f)
    assert(compileAndRun(unboxed[Double], Out1(F64(Float.MaxValue * 2.0))) === Float.MaxValue * 2.0)
  }

  @Test
  def _if() {
    assert(compileAndRun(unboxed[Int], If(True(), Out1(I32(10)), Out1(I32(-10)))) === 10)
    assert(compileAndRun(unboxed[Int], If(False(), Out1(I32(10)), Out1(I32(-10)))) === -10)
  }

  @Test
  def nestedIf() {
    assert(compileAndRun(unboxed[Int], If(True(), If(True(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 3)
    assert(compileAndRun(unboxed[Int], If(True(), If(False(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 2)
    assert(compileAndRun(unboxed[Int], If(False(), If(True(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 0)
  }

  @Test
  def let() {
    assert(compileAndRun(unboxed[Int], Let("x", I32(10), IntInfo, Out1(Ref("x")))) === 10)
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
  def simpleNullness() {
    assert(compileAndRun(boxed[Integer], Out1(Null())) === null)
    assert(compileAndRun(boxed[Integer], Out1(Null())) === null)
    assert(compileAndRun(boxed[Integer], If(True(), Out1(Null()), Out1(Null()))) === null)
    assert(compileAndRun(boxed[Integer], If(False(), Out1(Null()), Out1(Null()))) === null)
    assert(compileAndRun(boxed[Integer], Let("x", Null(), classInfo[Integer], Out1(Ref("x")))) === null)
    assert(compileAndRun(unboxed[Int], Let("x", Null(), classInfo[Integer], Out1(I32(1)))) === 1)
    assert(compileAndRun(boxed[Integer], Let("x", I32(2), IntInfo, Out1(Null()))) === null)
    assert(compileAndRun(boxed[Integer], Let("x", I32(2), IntInfo, Out1(Null()))) === null)
  }

  @Test
  def mapNull() {
    assert(compileAndRun(boxed[Integer], MapNull("x", Null(), classInfo[Integer], Out1(I32(1)))) === null)
    assert(compileAndRun(boxed[Integer], MapNull("x", Null(), classInfo[Integer], Out1(Ref("x")))) === null)
    assert(compileAndRun(boxed[Integer], MapNull("x", I32(3), classInfo[Integer], Out1(Ref("x")))) === null)
  }

  @Test
  def differingNullnessOnBranchesOfAnIf() {
    assert(compileAndRun(boxed[Integer], If(False(), Out1(Null()), Out1(I32(3)))) === 3)
  }

}
