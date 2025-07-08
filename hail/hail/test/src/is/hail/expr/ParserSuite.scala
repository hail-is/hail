package is.hail.expr

import is.hail.HailSuite

import org.scalatest
import org.testng.annotations.Test

class ParserSuite extends HailSuite {
  @Test def testOneOfLiteral(): scalatest.Assertion = {
    val strings = Array("A", "B", "AB", "AA", "CAD", "EF")
    val p = Parser.oneOfLiteral(strings)
    strings.foreach(s => assert(p.parse(s) == s))

    assert(p.parseOpt("hello^&").isEmpty)
    assert(p.parseOpt("ABhello").isEmpty)

    assert(Parser.rep(p).parse("ABCADEF") == List("AB", "CAD", "EF"))
  }
}
