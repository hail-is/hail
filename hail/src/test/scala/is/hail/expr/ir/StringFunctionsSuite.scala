package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import org.json4s.jackson.JsonMethods

import org.testng.annotations.{DataProvider, Test}

class StringFunctionsSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def testRegexMatch() {
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), NA(TString)), null)
    assertEvalsTo(invoke("regexMatch", TBoolean, NA(TString), Str("b")), null)

    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), Str("b")), false)
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), Str("a")), true)

    assertEvalsTo(invoke("regexMatch", TBoolean, Str("[a-z][0-9]"), Str("t7")), true)
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("[a-z][0-9]"), Str("3x")), false)
  }

  @Test def testLength() {
    assertEvalsTo(invoke("length", TInt32, Str("ab")), 2)
    assertEvalsTo(invoke("length", TInt32, Str("")), 0)
    assertEvalsTo(invoke("length", TInt32, NA(TString)), null)
  }

  @Test def testSubstring() {
    assertEvalsTo(invoke("substring", TString, Str("ab"), 0, 1), "a")
    assertEvalsTo(invoke("substring", TString, Str("ab"), NA(TInt32), 1), null)
    assertEvalsTo(invoke("substring", TString, Str("ab"), 0, NA(TInt32)), null)
    assertEvalsTo(invoke("substring", TString, NA(TString), 0, 1), null)
  }

  @Test def testConcat() {
    assertEvalsTo(invoke("concat", TString, Str("a"), NA(TString)), null)
    assertEvalsTo(invoke("concat", TString, NA(TString), Str("b")), null)

    assertEvalsTo(invoke("concat", TString, Str("a"), Str("b")), "ab")
  }

  @Test def testSplit() {
    assertEvalsTo(invoke("split", TArray(TString), NA(TString), Str(",")), null)
    assertEvalsTo(invoke("split", TArray(TString), Str("a,b,c"), NA(TString)), null)

    assertEvalsTo(invoke("split", TArray(TString), Str("x"), Str("x")), FastSeq("", ""))
    assertEvalsTo(invoke("split", TArray(TString), Str("a,b,c"), Str(",")), FastSeq("a", "b", "c"))

    assertEvalsTo(invoke("split", TArray(TString), NA(TString), Str(","), I32(2)), null)
    assertEvalsTo(invoke("split", TArray(TString), Str("a,b,c"), NA(TString), I32(2)), null)
    assertEvalsTo(invoke("split", TArray(TString), Str("a,b,c"), Str(","), NA(TInt32)), null)

    assertEvalsTo(
      invoke("split", TArray(TString), Str("a,b,c"), Str(","), I32(2)),
      FastSeq("a", "b,c"),
    )
  }

  @Test def testReplace() {
    assertEvalsTo(invoke("replace", TString, NA(TString), Str(","), Str(".")), null)
    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), NA(TString), Str(".")), null)
    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), Str(","), NA(TString)), null)

    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), Str(","), Str(".")), "a.b.c")
  }

  @Test def testArrayMkString() {
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", "b", "c"), NA(TString)), null)
    assertEvalsTo(invoke("mkString", TString, NA(TArray(TString)), Str(",")), null)
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", null, "c"), Str(",")), "a,null,c")
  }

  @Test def testSetMkString() {
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", "b", "c"), NA(TString)), null)
    assertEvalsTo(invoke("mkString", TString, NA(TSet(TString)), Str(",")), null)
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", null, "c"), Str(",")), "a,c,null")
  }

  @Test def testFirstMatchIn() {
    assertEvalsTo(invoke("firstMatchIn", TArray(TString), Str("""([a-zA-Z]+)"""), Str("1")), null)
    assertEvalsTo(
      invoke("firstMatchIn", TArray(TString), Str("Hello world!"), Str("""([a-zA-Z]+)""")),
      FastSeq("Hello"),
    )
    assertEvalsTo(
      invoke("firstMatchIn", TArray(TString), Str("Hello world!"), Str("""[a-zA-Z]+""")),
      FastSeq(),
    )
  }

  @Test def testHammingDistance() {
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), NA(TString)), null)
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), Str("fool")), null)
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), Str("fol")), 1)
  }

  @DataProvider(name = "str")
  def strData(): Array[Array[Any]] = Array(
    Array(NA(TString), TString),
    Array(NA(TStruct("x" -> TInt32)), TStruct("x" -> TInt32)),
    Array(F32(3.14f), TFloat32),
    Array(I64(7), TInt64),
    Array(IRArray(1, null, 5), TArray(TInt32)),
    Array(MakeTuple.ordered(FastSeq(1, NA(TInt32), 5.7)), TTuple(TInt32, TInt32, TFloat64)),
  )

  @Test(dataProvider = "str")
  def str(annotation: IR, typ: Type) {
    assertEvalsTo(
      invoke("str", TString, annotation), {
        val a = eval(annotation); if (a == null) null else typ.str(a)
      },
    )
  }

  @Test(dataProvider = "str")
  def json(annotation: IR, typ: Type) {
    assertEvalsTo(
      invoke("json", TString, annotation),
      JsonMethods.compact(typ.toJSON(eval(annotation))),
    )
  }

  @DataProvider(name = "time")
  def timeData(): Array[Array[Any]] = Array(
    // □ = untested
    // ■ = tested
    // ⊗ = unimplemented

    // % A a B b C c D d e F G g H I j k l M m n p R r S s T t U u V v W w X x Y y Z z
    // ■ ■ ■ ■ ■ ⊗ ⊗ ■ ■ ■ ■ ⊗ ⊗ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⊗ ⊗ ⊗ ■ ■ ⊗ ■

    Array("%t%%%n%s", "\t%\n123456789", 123456789),
    Array("%m/%d/%y %I:%M:%S %p", "10/10/97 11:45:23 PM", 876541523),
    Array("%m/%d/%y %I:%M:%S %p", "07/08/19 03:00:01 AM", 1562569201),
    Array("%Y.%m.%d %H:%M:%S %z", "1997.10.10 23:45:23 -04:00", 876541523),
    Array("%Y.%m.%d %H:%M:%S %Z", "2019.07.08 03:00:01 America/New_York", 1562569201),
    Array("day %j of %Y. %R:%S", "day 283 of 1997. 23:45:23", 876541523),
    Array("day %j of %Y. %R:%S", "day 189 of 2019. 03:00:01", 1562569201),
    Array("day %j of %Y. %R:%S", "day 001 of 1970. 22:46:40", 100000),
    Array("%v %T", "10-Oct-1997 23:45:23", 876541523),
    Array("%v %T", " 8-Jul-2019 03:00:01", 1562569201),
    Array("%A, %B %e, %Y. %r", "Friday, October 10, 1997. 11:45:23 PM", 876541523),
    Array("%A, %B %e, %Y. %r", "Monday, July  8, 2019. 03:00:01 AM", 1562569201),
    Array("%a, %b %e, '%y. %I:%M:%S %p", "Fri, Oct 10, '97. 11:45:23 PM", 876541523),
    Array("%a, %b %e, '%y. %I:%M:%S %p", "Mon, Jul  8, '19. 03:00:01 AM", 1562569201),
    Array("%D %l:%M:%S %p", "10/10/97 11:45:23 PM", 876541523),
    Array("%D %l:%M:%S %p", "07/08/19  3:00:01 AM", 1562569201),
    Array("%F %k:%M:%S", "1997-10-10 23:45:23", 876541523),
    Array("%F %k:%M:%S", "2019-07-08  3:00:01", 1562569201),
    Array(
      "ISO 8601 week day %u. %Y.%m.%d %H:%M:%S",
      "ISO 8601 week day 4. 1970.01.01 22:46:40",
      100000,
    ),
    Array(
      "Week number %U of %Y. %Y.%m.%d %H:%M:%S",
      "Week number 00 of 1973. 1973.01.01 10:33:20",
      94750400,
    ),
    Array(
      "ISO 8601 week #%V. %Y.%m.%d %H:%M:%S",
      "ISO 8601 week #53. 2005.01.02 00:00:00",
      1104642000,
    ),
    Array(
      "ISO 8601 week #%V. %Y.%m.%d %H:%M:%S",
      "ISO 8601 week #01. 2005.01.03 00:00:00",
      1104728400,
    ),
    Array("Monday week #%W. %Y.%m.%d %H:%M:%S", "Monday week #00. 2005.01.02 00:00:00", 1104642000),
    Array("Monday week #%W. %Y.%m.%d %H:%M:%S", "Monday week #01. 2005.01.03 00:00:00", 1104728400),
  )

  @Test(dataProvider = "time")
  def strftime(fmt: String, s: String, t: Long) {
    assertEvalsTo(invoke("strftime", TString, Str(fmt), I64(t), Str("America/New_York")), s)
  }

  @Test(dataProvider = "time")
  def strptime(fmt: String, s: String, t: Long) {
    assertEvalsTo(invoke("strptime", TInt64, Str(s), Str(fmt), Str("America/New_York")), t)
  }
}
