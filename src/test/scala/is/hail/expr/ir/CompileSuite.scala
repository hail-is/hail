package is.hail.methods.ir

import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.ir._
import is.hail.expr.ir.IR._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class CompileSuite {

  def compileAndRun[T: TypeInfo](ir: IR): T = {
    val fb = FunctionBuilder.functionBuilder[T]
    Compile()(ir).run(fb).apply()
  }

  @Test
  def constants() {
    assert(compileAndRun[Int](Out1(I32(16))) === 16)
    assert(compileAndRun[Long](Out1(I64(Int.MaxValue+1))) === Int.MaxValue+1)
    assert(compileAndRun[Float](Out1(F32(0f))) === 0f)
    assert(compileAndRun[Double](Out1(F64(Float.MaxValue * 2.0))) === Float.MaxValue * 2.0)
  }

  @Test
  def _if() {
    assert(compileAndRun[Int](_If(True(), Out1(I32(10)), Out1(I32(-10)))) === 10)
    assert(compileAndRun[Int](_If(False(), Out1(I32(10)), Out1(I32(-10)))) === -10)
  }

  @Test
  def nestedIf() {
    assert(compileAndRun[Int](_If(True(), _If(True(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 3)
    assert(compileAndRun[Int](_If(True(), _If(False(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 2)
    assert(compileAndRun[Int](_If(False(), _If(True(), Out1(I32(3)), Out1(I32(2))), Out1(I32(0)))) === 0)
  }

  @Test
  def let() {
    assert(compileAndRun[Int](Let("x", I32(10), IntInfo, Out1(Ref("x")))) === 10)
  }

  @Test
  def arrays() {
    assert(compileAndRun[Array[Int]](Out1(MakeArray(Array(), IntInfo))) === Array[Int]())
    assert(compileAndRun[Array[Int]](Out1(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo))) === Array(4,3,2,1,0))
    assert(compileAndRun[Int](Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(0), IntInfo))) === 4)
    assert(compileAndRun[Int](Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(4), IntInfo))) === 0)
    intercept[IndexOutOfBoundsException] {
      compileAndRun[Int](Out1(ArrayRef(MakeArray(Array(I32(4), I32(3), I32(2), I32(1), I32(0)), IntInfo), I32(5), IntInfo)))
    }
  }

}
