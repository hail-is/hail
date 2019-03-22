package is.hail.expr.ir

import is.hail.{ExecStrategy, SparkSuite}
import is.hail.TestUtils._
import org.testng.annotations.Test

class StringLengthSuite extends SparkSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def sameAsJavaStringLength() {
    val strings = Array("abc", "", "\uD83D\uDCA9")
    for (s <- strings) {
      assertEvalsTo(invoke("length", Str(s)), s.length)
    }
  }
}
