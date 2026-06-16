package is.hail.utils.prettyPrint

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterator

import org.junit.jupiter.api.Test

class PrettyPrintWriterSuite {
  private def data: Array[(Doc, Array[(Int, Int, Int, String)])] =
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

  def testPP() = ArraySeq[(Doc, Integer, Integer, Integer, String)](
    (for {
      (doc, cases) <- data.iterator
      (width, ribbonWidth, maxLines, expected) <- cases.iterator
    } yield (doc, Int.box(width), Int.box(ribbonWidth), Int.box(maxLines), expected)).toSeq: _*
  )

  @ParameterizedTest
  def testPP(doc: Doc, width: Integer, ribbonWidth: Integer, maxLines: Integer, expected: String)
    : Unit = {
    val ruler = "=" * width
    assertEq(s"$ruler\n${doc.render(width, ribbonWidth, maxLines)}\n$ruler", expected)
  }

  @Test def testIntersperse(): Unit = {
    val it = Array("A", "B", "C").iterator.intersperse("(", ",", ")")
    assertEq(it.mkString, "(A,B,C)")
  }

}
