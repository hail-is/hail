package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._

import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class DictFunctionsSuite extends TestNGSuite {
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

  @Test def toDict() {
    eval(a0)
    eval(d0)
    assertEvalsTo(d0, Map((1, 3), (2, 7)))
    assertEvalsTo(d, Map((1, 3), (2, null), (null, 1), (3, 7)))
    assertEvalsTo(nad, null)
    assertEvalsTo(ToDict(naa), null)
    assertEvalsTo(e, Map())
  }
}
