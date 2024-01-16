package is.hail.lir

import is.hail.HailSuite
import is.hail.asm4s._

import org.testng.annotations.Test

class CompileTimeRequirednessSuite extends HailSuite {
  @Test def testCodeBooleanFolding(): Unit = {
    val cFalse = const(false)
    val cTrue = const(true)

    assert(Code.constBoolValue(cFalse).contains(false))
    assert(Code.constBoolValue(cTrue).contains(true))
    assert(Code.constBoolValue(!cFalse).contains(true))
    assert(Code.constBoolValue(!cTrue).contains(false))
    assert(Code.constBoolValue(const(1) < 2).isEmpty)
  }
}
