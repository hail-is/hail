package is.hail.utils

import is.hail.HailSuite
import is.hail.expr.ir.TextTableReader
import is.hail.expr.types.virtual._
import org.testng.annotations.Test

class TextTableSuite extends HailSuite {

  @Test def testTypeGuessing() {

    val doubleStrings = Seq("1", ".1", "-1", "-.1", "1e1", "-1e1",
      "1E1", "-1E1", "1.0e2", "-1.0e2", "1e-1", "-1e-1", "-1.0e-2")
    val badDoubleStrings = Seq("1ee1", "1e--2", "1ee2", "1e0.1", "1e-0.1", "1e1.")
    val intStrings = Seq("1", "0", "-1", "12312398", "-123092398")
    val longStrings = Seq("11010101010101010", "-9223372036854775808")
    val booleanStrings = Seq("true", "True", "TRUE", "false", "False", "FALSE")

    doubleStrings.foreach(str => assert(TextTableReader.float64Matcher(str)))
    badDoubleStrings.foreach(str => assert(!TextTableReader.float64Matcher(str)))

    intStrings.foreach(str => assert(TextTableReader.int32Matcher(str)))
    intStrings.foreach(str => assert(TextTableReader.int64Matcher(str)))
    intStrings.foreach(str => assert(TextTableReader.float64Matcher(str)))

    longStrings.foreach(str => assert(TextTableReader.int64Matcher(str), str))
    longStrings.foreach(str => assert(!TextTableReader.int32Matcher(str)))

    booleanStrings.foreach(str => assert(TextTableReader.booleanMatcher(str)))
  }

  @Test def testPipeDelimiter() {
    assert(TextTableReader.splitLine("a|b", "|", '#').toSeq == Seq("a", "b"))
  }
}
