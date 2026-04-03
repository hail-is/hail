package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{F32, I32, I64, MakeTuple, NA, Str}
import is.hail.types.virtual._

import org.json4s.jackson.JsonMethods

class StringFunctionsSuite extends HailSuite {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  test("RegexMatch") {
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), NA(TString)), null)
    assertEvalsTo(invoke("regexMatch", TBoolean, NA(TString), Str("b")), null)

    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), Str("b")), false)
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("a"), Str("a")), true)

    assertEvalsTo(invoke("regexMatch", TBoolean, Str("[a-z][0-9]"), Str("t7")), true)
    assertEvalsTo(invoke("regexMatch", TBoolean, Str("[a-z][0-9]"), Str("3x")), false)
  }

  test("Length") {
    assertEvalsTo(invoke("length", TInt32, Str("ab")), 2)
    assertEvalsTo(invoke("length", TInt32, Str("")), 0)
    assertEvalsTo(invoke("length", TInt32, NA(TString)), null)
  }

  test("Substring") {
    assertEvalsTo(invoke("substring", TString, Str("ab"), 0, 1), "a")
    assertEvalsTo(invoke("substring", TString, Str("ab"), NA(TInt32), 1), null)
    assertEvalsTo(invoke("substring", TString, Str("ab"), 0, NA(TInt32)), null)
    assertEvalsTo(invoke("substring", TString, NA(TString), 0, 1), null)
  }

  test("Concat") {
    assertEvalsTo(invoke("concat", TString, Str("a"), NA(TString)), null)
    assertEvalsTo(invoke("concat", TString, NA(TString), Str("b")), null)

    assertEvalsTo(invoke("concat", TString, Str("a"), Str("b")), "ab")
  }

  test("Split") {
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

  test("Replace") {
    assertEvalsTo(invoke("replace", TString, NA(TString), Str(","), Str(".")), null)
    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), NA(TString), Str(".")), null)
    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), Str(","), NA(TString)), null)

    assertEvalsTo(invoke("replace", TString, Str("a,b,c"), Str(","), Str(".")), "a.b.c")
  }

  test("ArrayMkString") {
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", "b", "c"), NA(TString)), null)
    assertEvalsTo(invoke("mkString", TString, NA(TArray(TString)), Str(",")), null)
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", TString, IRStringArray("a", null, "c"), Str(",")), "a,null,c")
  }

  test("SetMkString") {
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", "b", "c"), NA(TString)), null)
    assertEvalsTo(invoke("mkString", TString, NA(TSet(TString)), Str(",")), null)
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", "b", "c"), Str(",")), "a,b,c")

    // FIXME matches current FunctionRegistry, but should be a,NA,c
    assertEvalsTo(invoke("mkString", TString, IRStringSet("a", null, "c"), Str(",")), "a,c,null")
  }

  test("FirstMatchIn") {
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

  test("HammingDistance") {
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), NA(TString)), null)
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), Str("fool")), null)
    assertEvalsTo(invoke("hamming", TInt32, Str("foo"), Str("fol")), 1)
  }

  val strData: Array[(IR, Type)] = Array(
    (NA(TString), TString),
    (NA(TStruct("x" -> TInt32)), TStruct("x" -> TInt32)),
    (F32(3.14f), TFloat32),
    (I64(7), TInt64),
    (IRArray(1, null, 5), TArray(TInt32)),
    (MakeTuple.ordered(FastSeq(1, NA(TInt32), 5.7)), TTuple(TInt32, TInt32, TFloat64)),
  )

  object checkStr extends TestCases {
    def apply(
      annotation: IR,
      typ: Type,
    )(implicit loc: munit.Location
    ): Unit = test("str") {
      assertEvalsTo(
        invoke("str", TString, annotation), {
          val a = eval(annotation); if (a == null) null else typ.str(a)
        },
      )
    }
  }

  strData.foreach { case (a, t) => checkStr(a, t) }

  object checkJson extends TestCases {
    def apply(
      annotation: IR,
      typ: Type,
    )(implicit loc: munit.Location
    ): Unit = test("json") {
      assertEvalsTo(
        invoke("json", TString, annotation),
        JsonMethods.compact(typ.export(eval(annotation))),
      )
    }
  }

  strData.foreach { case (a, t) => checkJson(a, t) }

  val timeData: Array[(String, String, Long)] = Array(
    // □ = untested
    // ■ = tested
    // ⊗ = unimplemented

    // % A a B b C c D d e F G g H I j k l M m n p R r S s T t U u V v W w X x Y y Z z
    // ■ ■ ■ ■ ■ ⊗ ⊗ ■ ■ ■ ■ ⊗ ⊗ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ⊗ ⊗ ⊗ ■ ■ ⊗ ■

    ("%t%%%n%s", "\t%\n123456789", 123456789L),
    ("%m/%d/%y %I:%M:%S %p", "10/10/97 11:45:23 PM", 876541523L),
    ("%m/%d/%y %I:%M:%S %p", "07/08/19 03:00:01 AM", 1562569201L),
    ("%Y.%m.%d %H:%M:%S %z", "1997.10.10 23:45:23 -04:00", 876541523L),
    ("%Y.%m.%d %H:%M:%S %Z", "2019.07.08 03:00:01 America/New_York", 1562569201L),
    ("day %j of %Y. %R:%S", "day 283 of 1997. 23:45:23", 876541523L),
    ("day %j of %Y. %R:%S", "day 189 of 2019. 03:00:01", 1562569201L),
    ("day %j of %Y. %R:%S", "day 001 of 1970. 22:46:40", 100000L),
    ("%v %T", "10-Oct-1997 23:45:23", 876541523L),
    ("%v %T", " 8-Jul-2019 03:00:01", 1562569201L),
    ("%A, %B %e, %Y. %r", "Friday, October 10, 1997. 11:45:23 PM", 876541523L),
    ("%A, %B %e, %Y. %r", "Monday, July  8, 2019. 03:00:01 AM", 1562569201L),
    ("%a, %b %e, '%y. %I:%M:%S %p", "Fri, Oct 10, '97. 11:45:23 PM", 876541523L),
    ("%a, %b %e, '%y. %I:%M:%S %p", "Mon, Jul  8, '19. 03:00:01 AM", 1562569201L),
    ("%D %l:%M:%S %p", "10/10/97 11:45:23 PM", 876541523L),
    ("%D %l:%M:%S %p", "07/08/19  3:00:01 AM", 1562569201L),
    ("%F %k:%M:%S", "1997-10-10 23:45:23", 876541523L),
    ("%F %k:%M:%S", "2019-07-08  3:00:01", 1562569201L),
    (
      "ISO 8601 week day %u. %Y.%m.%d %H:%M:%S",
      "ISO 8601 week day 4. 1970.01.01 22:46:40",
      100000L,
    ),
    (
      "Week number %U of %Y. %Y.%m.%d %H:%M:%S",
      "Week number 00 of 1973. 1973.01.01 10:33:20",
      94750400L,
    ),
    ("ISO 8601 week #%V. %Y.%m.%d %H:%M:%S", "ISO 8601 week #53. 2005.01.02 00:00:00", 1104642000L),
    ("ISO 8601 week #%V. %Y.%m.%d %H:%M:%S", "ISO 8601 week #01. 2005.01.03 00:00:00", 1104728400L),
    ("Monday week #%W. %Y.%m.%d %H:%M:%S", "Monday week #00. 2005.01.02 00:00:00", 1104642000L),
    ("Monday week #%W. %Y.%m.%d %H:%M:%S", "Monday week #01. 2005.01.03 00:00:00", 1104728400L),
  )

  object checkStrftime extends TestCases {
    def apply(
      fmt: String,
      s: String,
      t: Long,
    )(implicit loc: munit.Location
    ): Unit = test("strftime") {
      assertEvalsTo(invoke("strftime", TString, Str(fmt), I64(t), Str("America/New_York")), s)
    }
  }

  timeData.foreach { case (fmt, s, t) => checkStrftime(fmt, s, t) }

  object checkStrptime extends TestCases {
    def apply(
      fmt: String,
      s: String,
      t: Long,
    )(implicit loc: munit.Location
    ): Unit = test("strptime") {
      assertEvalsTo(invoke("strptime", TInt64, Str(s), Str(fmt), Str("America/New_York")), t)
    }
  }

  timeData.foreach { case (fmt, s, t) => checkStrptime(fmt, s, t) }
}
