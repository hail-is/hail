package is.hail.expr

import is.hail.HailSuite
import is.hail.collection.compat.immutable.ArraySeq

class ParserSuite extends HailSuite {
  test("OneOfLiteral") {
    val strings = ArraySeq("A", "B", "AB", "AA", "CAD", "EF")
    val p = Parser.oneOfLiteral(strings)
    strings.foreach(s => assertEquals(p.parse(s), s))

    assert(p.parseOpt("hello^&").isEmpty)
    assert(p.parseOpt("ABhello").isEmpty)

    assertEquals(Parser.rep(p).parse("ABCADEF"), List("AB", "CAD", "EF"))
  }
}
