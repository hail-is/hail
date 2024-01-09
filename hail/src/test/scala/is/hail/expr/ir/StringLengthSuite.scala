package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.types.virtual.TInt32

import org.testng.annotations.Test

class StringLengthSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def sameAsJavaStringLength() {
    val strings = Array("abc", "", "\uD83D\uDCA9")
    for (s <- strings)
      assertEvalsTo(invoke("length", TInt32, Str(s)), s.length)
  }
}
