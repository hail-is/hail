package is.hail.asm4s

import is.hail.HailSuite
import is.hail.expr.ir.EmitFunctionBuilder
import org.testng.annotations.Test

class CodeSuite extends HailSuite {

  @Test def testForLoop() {
    val fb = EmitFunctionBuilder[Int]("foo")
    val mb = fb.apply_method
    val i = mb.newLocal[Int]()
    val sum = mb.newLocal[Int]()
    val code = Code(
      sum := 0,
      Code.forLoop(i := 0, i < 5, i := i + 1, sum :=  sum + i),
      sum.load()
    )
    fb.emit(code)
    val result = fb.resultWithIndex()(0, ctx.r)()
    assert(result == 10)
  }
}
