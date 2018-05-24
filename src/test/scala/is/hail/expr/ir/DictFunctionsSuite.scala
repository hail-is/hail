package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class DictFunctionsSuite extends TestNGSuite {
  def IRArray(a: Integer*): IR = toIRArray(a)
  val naa = NA(TArray(TTuple(TInt32(), TInt32())))
  val a0 = MakeArray(Seq(MakeTuple(Seq(I32(1), I32(3))), MakeTuple(Seq(I32(2), I32(7)))), TArray(TTuple(TInt32(), TInt32())))
  val d0 = ToDict(a0)
  val a = MakeArray(
    Seq(
      MakeTuple(Seq(I32(1), I32(3))),
      MakeTuple(Seq(I32(1), I32(3))),
      MakeTuple(Seq(I32(2), NA(TInt32()))),
      NA(TTuple(TInt32(), TInt32())),
      MakeTuple(Seq(NA(TInt32()), I32(1))),
      MakeTuple(Seq(I32(3), I32(7)))),
    TArray(TTuple(TInt32(), TInt32())))
  val d = ToDict(a)
  val nad = NA(TDict(TInt32(), TInt32()))
  val e = ToDict(MakeArray(Seq(), TArray(TTuple(TInt32(), TInt32()))))

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
