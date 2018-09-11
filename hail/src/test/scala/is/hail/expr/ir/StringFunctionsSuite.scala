package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.utils.FastIndexedSeq
import org.json4s.jackson.JsonMethods
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class StringFunctionsSuite extends TestNGSuite {
  @Test def testRegexMatch() {
    assertEvalsTo(invoke("~", Str("a"), NA(TString())), null)
    assertEvalsTo(invoke("~", NA(TString()), Str("b")), null)

    assertEvalsTo(invoke("~", Str("a"), Str("b")), false)
    assertEvalsTo(invoke("~", Str("a"), Str("a")), true)

    assertEvalsTo(invoke("~", Str("[a-z][0-9]"), Str("t7")), true)
    assertEvalsTo(invoke("~", Str("[a-z][0-9]"), Str("3x")), false)
  }

  @Test def testConcat() {
    assertEvalsTo(invoke("+", Str("a"), NA(TString())), null)
    assertEvalsTo(invoke("+", NA(TString()), Str("b")), null)

    assertEvalsTo(invoke("+", Str("a"), Str("b")), "ab")
  }

  @Test def testLength() {
    assertEvalsTo(invoke("length", NA(TString())), null)
    assertEvalsTo(invoke("size", NA(TString())), null)

    assertEvalsTo(invoke("length", Str("five")), 4)
    assertEvalsTo(invoke("size", Str("five")), 4)
  }

  @Test def testSplit() {
    assertEvalsTo(invoke("split", NA(TString()), Str(",")), null)
    assertEvalsTo(invoke("split", Str("a,b,c"), NA(TString())), null)

    assertEvalsTo(invoke("split", Str("a,b,c"), Str(",")), FastIndexedSeq("a", "b", "c"))

    assertEvalsTo(invoke("split", NA(TString()), Str(","), I32(2)), null)
    assertEvalsTo(invoke("split", Str("a,b,c"), NA(TString()), I32(2)), null)
    assertEvalsTo(invoke("split", Str("a,b,c"), Str(","), NA(TInt32())), null)

    assertEvalsTo(invoke("split", Str("a,b,c"), Str(","), I32(2)), FastIndexedSeq("a", "b,c"))
  }

  @Test def testReplace() {
    assertEvalsTo(invoke("replace", NA(TString()), Str(","), Str(".")), null)
    assertEvalsTo(invoke("replace", Str("a,b,c"), NA(TString()), Str(".")), null)
    assertEvalsTo(invoke("replace", Str("a,b,c"), Str(","), NA(TString())), null)

    assertEvalsTo(invoke("replace", Str("a,b,c"), Str(","), Str(".")), "a.b.c")
  }

  @Test def testArrayMkString() {
    assertEvalsTo(invoke("mkString", IRStringArray("a", "b", "c"), NA(TString())), null)
    assertEvalsTo(invoke("mkString", NA(TArray(TString())), Str(",")), null)
    assertEvalsTo(invoke("mkString", IRStringArray("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", IRStringArray("a", null, "c"), Str(",")), "a,null,c")
  }

  @Test def testSetMkString() {
    assertEvalsTo(invoke("mkString", IRStringSet("a", "b", "c"), NA(TString())), null)
    assertEvalsTo(invoke("mkString", NA(TSet(TString())), Str(",")), null)
    assertEvalsTo(invoke("mkString", IRStringSet("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", IRStringSet("a", null, "c"), Str(",")), "a,c,null")

    @DataProvider(name="str")
    def strData(): Array[Array[Any]] = Array(
      Array(NA(TString()), TString()),
      Array(NA(TStruct("x" -> TInt32())), TStruct("x" -> TInt32())),
      Array(F32(3.14f), TFloat32()),
      Array(I64(7), TInt64()),
      Array(IRArray(1, null, 5), TArray(TInt32())),
      Array(MakeTuple(Seq(1, NA(TInt32()), 5.7)), TTuple(TInt32(), TInt32(), TFloat64()))
    )

    @Test(dataProvider = "str")
    def str(annotation: IR, typ: Type) {
      assertEvalsTo(invoke("str", annotation), typ.str(eval(annotation)))
    }

    @Test(dataProvider = "str")
    def json(annotation: IR, typ: Type) {
      assertEvalsTo(invoke("json", annotation), JsonMethods.compact(typ.toJSON(eval(annotation))))
    }
  }
}
