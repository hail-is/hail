package is.hail.utils.prettyPrint

import is.hail.TestCaseSupport
import is.hail.collection.implicits.toRichIterator

class PrettyPrintWriterSuite extends munit.FunSuite with TestCaseSupport {
  val data: Array[(Doc, Array[(Int, Int, Int, String)])] =
    Array(
      (
        nest(2, hsep("prefix", sep("text", "to", "lay", "out"))),
        Array(
          (
            25,
            25,
            5,
            """=========================
              |prefix text to lay out
              |=========================""".stripMargin,
          ),
          (
            20,
            20,
            5,
            """====================
              |prefix text
              |  to
              |  lay
              |  out
              |====================""".stripMargin,
          ),
          (
            20,
            20,
            2,
            """====================
              |prefix text
              |  to
              |====================""".stripMargin,
          ),
        ),
      ),
      (
        nest(2, hsep("prefix", fillSep("text", "to", "lay", "out"))),
        Array(
          (
            25,
            25,
            5,
            """=========================
              |prefix text to lay out
              |=========================""".stripMargin,
          ),
          (
            20,
            20,
            5,
            """====================
              |prefix text to lay
              |  out
              |====================""".stripMargin,
          ),
          (
            15,
            15,
            5,
            """===============
              |prefix text to
              |  lay out
              |===============""".stripMargin,
          ),
          (
            10,
            10,
            5,
            """==========
              |prefix text
              |  to lay
              |  out
              |==========""".stripMargin,
          ),
          (
            10,
            10,
            2,
            """==========
              |prefix text
              |  to lay
              |==========""".stripMargin,
          ),
        ),
      ),
      (
        nest(2, concat("prefix", list("A", "B", "C", "D"))),
        Array(
          (
            15,
            15,
            5,
            """===============
              |prefix(A B C D)
              |===============""".stripMargin,
          ),
          (
            10,
            10,
            5,
            """==========
              |prefix(A
              |  B
              |  C
              |  D)
              |==========""".stripMargin,
          ),
          (
            10,
            10,
            3,
            """==========
              |prefix(A
              |  B
              |  C
              |==========""".stripMargin,
          ),
        ),
      ),
    )

  object checkPP extends TestCases {
    def apply(
      doc: Doc,
      width: Int,
      ribbonWidth: Int,
      maxLines: Int,
      expected: String,
    )(implicit loc: munit.Location
    ): Unit = test("PP") {
      val ruler = "=" * width
      assertEquals(expected, s"$ruler\n${doc.render(width, ribbonWidth, maxLines)}\n$ruler")
    }
  }

  for {
    (doc, cases) <- data
    (width, ribbonWidth, maxLines, expected) <- cases
  } checkPP(doc, width, ribbonWidth, maxLines, expected)

  test("Intersperse") {
    val it = Array("A", "B", "C").iterator.intersperse("(", ",", ")")
    assertEquals(it.mkString, "(A,B,C)")
  }
}
