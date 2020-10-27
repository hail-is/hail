package is.hail.utils.prettyPrint

import is.hail.utils.toRichIterator

import scala.collection.JavaConverters._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class PrettyPrintWriterSuite extends TestNGSuite {
  def data: Array[(Doc, Array[(Int, Int, String)])] =
    Array(
      ( nest(2, hsep("prefix", sep("text", "to", "lay", "out"))),
        Array(
          (25, 25,
            """=========================
              |prefix text to lay out
              |=========================""".stripMargin),
          (20, 20,
            """====================
              |prefix text
              |  to
              |  lay
              |  out
              |====================""".stripMargin))),
      ( nest(2, concat("prefix", list("A", "B", "C", "D"))),
        Array(
          (20, 20,
            """====================
              |prefix(A, B, C, D)
              |====================""".stripMargin),
          (15, 15,
            """===============
              |prefix(
              |  A,
              |  B,
              |  C,
              |  D)
              |===============""".stripMargin)))
    )

  @DataProvider(name = "data")
  def flatData: java.util.Iterator[Array[Object]] =
    (for {
      (doc, cases) <- data.iterator
      (width, ribbonWidth, expected) <- cases.iterator
    } yield Array(doc, new Integer(width), new Integer(ribbonWidth), expected)).asJava

  @Test(dataProvider = "data")
  def testPP(doc: Doc, width: Integer, ribbonWidth: Integer, expected: String): Unit = {
    val ruler = "=" * width
    assert(expected == s"$ruler\n${ doc.render(width, ribbonWidth) }\n$ruler")
  }

  @Test def testIntersperse() {
    Array("A", "B", "C").iterator.intersperse("(", ",", ")").foreach(println)
  }

}
