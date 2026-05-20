package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.defs.Str
import is.hail.types.virtual.TInt32

import org.junit.jupiter.api.Test

class StringLengthSuite {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  @Test def sameAsJavaStringLength(implicit ctx: ExecuteContext): Unit = {
    val strings = Array("abc", "", "\uD83D\uDCA9")
    for (s <- strings)
      assertEvalsTo(invoke("length", TInt32, Str(s)), s.length)
  }
}
