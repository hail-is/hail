package is.hail.expr.ir

import is.hail.expr.types.{TArray, TSet, TString}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class StringFunctionsSuite extends TestNGSuite {
  @Test def testArrayMkString() {
    assertEvalsTo(invoke("mkString", NA(TArray(TString())), NA(TString())), null)
    assertEvalsTo(invoke("mkString", IRStringArray("a", "b", "c"), NA(TString())), null)
    assertEvalsTo(invoke("mkString", NA(TArray(TString())), Str(",")), null)
    assertEvalsTo(invoke("mkString", IRStringArray("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", IRStringArray("a", null, "c"), Str(",")), "a,null,c")
  }

  @Test def testSetMkString() {
    assertEvalsTo(invoke("mkString", NA(TSet(TString())), NA(TString())), null)
    assertEvalsTo(invoke("mkString", IRStringSet("a", "b", "c"), NA(TString())), null)
    assertEvalsTo(invoke("mkString", NA(TSet(TString())), Str(",")), null)
    assertEvalsTo(invoke("mkString", IRStringSet("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", IRStringSet("a", null, "c"), Str(",")), "a,c,null")
  }
}
