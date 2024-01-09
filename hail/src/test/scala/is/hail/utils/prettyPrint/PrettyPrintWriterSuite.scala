package is.hail.utils.prettyPrint

import is.hail.utils.toRichIterator

import scala.collection.JavaConverters._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class PrettyPrintWriterSuite extends TestNGSuite {
  def data: Array[(Doc, Array[(Int, Int, Int, String)])] =
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

  @DataProvider(name = "data")
  def flatData: java.util.Iterator[Array[Object]] =
    (for {
      (doc, cases) <- data.iterator
      (width, ribbonWidth, maxLines, expected) <- cases.iterator
    } yield Array(doc, Int.box(width), Int.box(ribbonWidth), Int.box(maxLines), expected)).asJava

  @Test(dataProvider = "data")
  def testPP(doc: Doc, width: Integer, ribbonWidth: Integer, maxLines: Integer, expected: String)
    : Unit = {
    val ruler = "=" * width
    assert(expected == s"$ruler\n${doc.render(width, ribbonWidth, maxLines)}\n$ruler")
  }

  @Test def testIntersperse() {
    val it = Array("A", "B", "C").iterator.intersperse("(", ",", ")")
    assert(it.mkString == "(A,B,C)")
  }

}
