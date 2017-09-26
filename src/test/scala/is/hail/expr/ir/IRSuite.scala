package is.hail.methods.ir

import is.hail.asm4s._
import is.hail.check.{Gen, Parameters, Prop}
import is.hail.expr.TInt32
import is.hail.expr.ir._
import is.hail.expr.ir.IR._
import org.testng.annotations.Test
import org.scalatest._
import Matchers._

class IRSuite {
  @Test
  def simpleUsedInputs() {
    assert(usedInputs(In(1)) === Array(1))
    assert(usedInputs(In(2)) === Array(2))
    assert(usedInputs(In(100)) === Array(100))
  }

  @Test
  def nestedUsedInputs() {
    assert(usedInputs(Let("foo", I32(1), TInt32, In(1))) === Array(1))
    // FIXME: this is naively used, but not really used
    assert(usedInputs(Let("foo", In(1), TInt32, I32(1))) === Array(1))
    assert(usedInputs(Out(Array(In(1), In(2)))) === Array(1, 2))
    assert(usedInputs(ApplyPrimitive(PrimitiveOp(0), Array(In(1), In(2))))
      === Array(1, 2))
  }

  @Test(enabled = false)
  def smarterNestedUsedInputs() {
    assert(usedInputs(Let("foo", In(1), TInt32, I32(1))) === Array())
  }

  @Test
  def noUsedInputs() {
    assert(usedInputs(I32(1)) === Array())
    assert(usedInputs(F64(1.0)) === Array())
    assert(usedInputs(Let("foo", I32(1), TInt32, Ref("foo"))) === Array())
  }
}
