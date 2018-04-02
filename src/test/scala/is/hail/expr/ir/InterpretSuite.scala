package is.hail.expr.ir

import is.hail.expr.types._
import org.junit.Test

class InterpretSuite {


  @Test def test() {
    val env = Env.empty[(Any, Type)]
      .bind("a", 1.5 -> TFloat64Optional)
      .bind("b", 3.0 -> TFloat64Required)

    assert(Interpret(ApplyBinaryPrimOp(Add(), Ref("a"), Ref("b")), env, IndexedSeq(), None) == 4.5)
  }
}
