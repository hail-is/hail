package is.hail.expr

import is.hail.SparkSuite
import org.testng.annotations.Test

class ParserSuite extends SparkSuite{
  @Test def testOneOfLiteral() = {
    val strings = Array("A", "B", "AB", "AA", "CAD", "EF")
    val p = Parser.oneOfLiteral(strings)
    strings.foreach(s => assert(p.parse(s) == s))

    assert(p.parse_opt("hello^&").isEmpty)
    assert(p.parse_opt("ABhello").isEmpty)

    assert(Parser.rep(p).parse("ABCADEF") == List("AB", "CAD", "EF"))
  }
}
