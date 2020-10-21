package is.hail.utils.prettyPrint

import is.hail.utils.toRichIterator

import scala.collection.JavaConverters._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class PrettyPrintWriterSuite extends TestNGSuite {
  import PrettyPrintWriter._

  def data: Array[(() => Doc, Array[(Int, String)])] =
    Array(
      ( () => nest(2, hsep("prefix", sep("text", "to", "lay", "out"))),
        Array(
          25 ->
            """=========================
              |prefix text to lay out
              |=========================""".stripMargin,
          20 ->
            """====================
              |prefix text
              |  to
              |  lay
              |  out
              |====================""".stripMargin)),
      ( () => nest(2, concat("prefix", list("A", "B", "C", "D"))),
        Array(
          20 ->
            """====================
              |prefix(A, B, C, D)
              |====================""".stripMargin,
          18 ->
            """==================
              |prefix(
              |  A,
              |  B,
              |  C,
              |  D)
              |==================""".stripMargin))
    )

  @DataProvider(name = "data")
  def flatData: java.util.Iterator[Array[Object]] =
    (for {
      (doc, cases) <- data.iterator
      (width, expected) <- cases.iterator
    } yield Array(doc, new Integer(width), expected)).asJava

  @Test(dataProvider = "data")
  def testPP(doc: () => Doc, width: Integer, expected: String): Unit = {
    val ruler = "=" * width
    assert(expected == s"$ruler\n${ doc().render(width) }\n$ruler")
  }

  @Test def testIntersperse() {
    Array("A", "B", "C").iterator.intersperse("(", ",", ")").foreach(println)
  }

}
