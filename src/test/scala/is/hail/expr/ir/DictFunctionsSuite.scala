package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class DictFunctionsSuite extends TestNGSuite {

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(Seq((1, 3), (2, 7))),
    Array(Seq((1, 3), (2, null), null, (null, 1), (3, 7))),
    Array(Seq()),
    Array(Seq(null)),
    Array(null)
  )

  @Test(dataProvider = "basic")
  def toDict(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("toDict", toIRPairArray(a)),
      Option(a).map(_.filter(_ != null).toMap).orNull)
    assertEvalsTo(toIRDict(a),
      Option(a).map(_.filter(_ != null).toMap).orNull)
  }

  @Test(dataProvider = "basic")
  def size(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("size", toIRDict(a)),
      Option(a).map(_.count(_ != null)).orNull)
  }

  @Test(dataProvider = "basic")
  def isEmpty(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("isEmpty", toIRDict(a)),
      Option(a).map(_.forall(_ == null)).orNull)
  }
}
