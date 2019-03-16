package is.hail.expr.ir

import is.hail.{ExecStrategy, SparkSuite}
import is.hail.TestUtils._
import is.hail.expr.types._
import is.hail.expr.types.virtual.TString
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class StringLengthSuite extends SparkSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def sameAsJavaStringLength() {
    val strings = Array("abc", "", new String(Array[Char](0xD83D, 0xDCA9)))
    for (s <- strings) {
      assertEvalsTo(invoke("length", Str(s)), s.length)
    }
  }
}
