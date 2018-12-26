package is.hail.expr.ir

import is.hail.expr.ir.functions.AnonymizeBindings
import is.hail.expr.types.virtual.TInt32
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class AnonymizeBindingsSuite extends TestNGSuite {
  @Test def testNestedLets(): Unit = {
    val ir = Let(
      "x",
      I32(1),
      Let(
        "x",
        I32(2),
        Ref("x", TInt32())
      )
    )
    val Let(name1, _, Let(name2, _, Ref(name3, _))) = AnonymizeBindings(ir)
    assert(name1 != name2)
    assert(name2 == name3)
  }
}
