package is.hail.expr

import is.hail.collection.compat.immutable.ArraySeq

import org.junit.jupiter.api.Test

class ParserSuite {
  @Test def testOneOfLiteral(): Unit = {
    val strings = ArraySeq("A", "B", "AB", "AA", "CAD", "EF")
    val p = Parser.oneOfLiteral(strings)
    strings.foreach(s => assert(p.parse(s) == s))

    assert(p.parseOpt("hello^&").isEmpty)
    assert(p.parseOpt("ABhello").isEmpty)

    assert(Parser.rep(p).parse("ABCADEF") == List("AB", "CAD", "EF"))
  }
}
