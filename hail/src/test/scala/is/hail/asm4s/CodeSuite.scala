package is.hail.asm4s

import is.hail.HailSuite
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.interfaces.{SString, SStringCode}
import is.hail.types.physical.stypes.primitives.SInt32Code
import org.testng.annotations.Test

class CodeSuite extends HailSuite {

  @Test def testForLoop() {
    val fb = EmitFunctionBuilder[Int](ctx, "foo")
    val mb = fb.apply_method
    val i = mb.newLocal[Int]()
    val sum = mb.newLocal[Int]()
    val code = Code(
      sum := 0,
      Code.forLoop(i := 0, i < 5, i := i + 1, sum :=  sum + i),
      sum.load()
    )
    fb.emit(code)
    val result = fb.resultWithIndex()(ctx.fs, 0, ctx.r)()
    assert(result == 10)
  }
  def hashTestHelper(toHash: SCode): Int = {
    val fb = EmitFunctionBuilder[Int](ctx, "test_hash")
    val mb = fb.apply_method
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val i = toHash.memoize(cb, "int_to_hash")
      val hash = i.hash(cb)
      hash.intCode(cb)
    })
    fb.result()()()
  }
  @Test def testHash() {
    assert(hashTestHelper(new SInt32Code(6)) == hashTestHelper(new SInt32Code(6)))
    assert(hashTestHelper(new SStringCode(const("dog"))) == hashTestHelper(new SString(6)))
  }
}
