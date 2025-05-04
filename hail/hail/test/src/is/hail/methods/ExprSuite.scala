package is.hail.methods

import is.hail.HailSuite
import is.hail.backend.HailStateManager
import is.hail.expr._
import is.hail.expr.ir.IRParser
import is.hail.scalacheck._
import is.hail.types.virtual._
import is.hail.utils.StringEscapeUtils._

import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.scalacheck.Arbitrary._
import org.scalacheck.Gen
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.testng.annotations.Test

class ExprSuite extends HailSuite with ScalaCheckDrivenPropertyChecks {

  def sm: HailStateManager = ctx.stateManager

  @Test def testTypePretty(): Unit = {
    // for arbType

    val sb = new StringBuilder
    forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, 0, compact = true)
      val res = sb.result()
      val parsed = IRParser.parseType(res)
      t == parsed
    }

    forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, 0, compact = false)
      val res = sb.result()
      val parsed = IRParser.parseType(res)
      t == parsed
    }

    forAll { (t: Type) =>
      val s = t.parsableString()
      val parsed = IRParser.parseType(s)
      t == parsed
    }
  }

  @Test def testEscaping(): Unit =
    forAll((s: String) => s == unescapeString(escapeString(s)))

  @Test def testEscapingSimple(): Unit = {
    // a == 0x61, _ = 0x5f
    assert(escapeStringSimple("abc", '_', _ => false) == "abc")
    assert(escapeStringSimple("abc", '_', _ == 'a') == "_61bc")
    assert(escapeStringSimple("abc_", '_', _ => false) == "abc_5f")
    assert(unescapeStringSimple("abc", '_') == "abc")
    assert(unescapeStringSimple("abc_5f", '_') == "abc_")
    assert(unescapeStringSimple("_61bc", '_') == "abc")
    assert(unescapeStringSimple("_u0061bc", '_') == "abc")
    assert(escapeStringSimple("my name is 名谦", '_', _ => false) == "my name is _u540d_u8c26")
    assert(unescapeStringSimple("my name is _u540d_u8c26", '_') == "my name is 名谦")

    forAll(Gen.asciiPrintableStr) { (s: String) =>
      s == unescapeStringSimple(
        escapeStringSimple(s, '_', _.isLetterOrDigit, _.isLetterOrDigit),
        '_',
      )
    }
  }

  @Test def testImportEmptyJSONObjectAsStruct(): Unit =
    assert(JSONAnnotationImpex.importAnnotation(parse("{}"), TStruct()) == Row())

  @Test def testExportEmptyJSONObjectAsStruct(): Unit =
    assert(compact(render(JSONAnnotationImpex.exportAnnotation(Row(), TStruct()))) == "{}")

  @Test def testRoundTripEmptyJSONObject(): Unit = {
    val actual = JSONAnnotationImpex.exportAnnotation(
      JSONAnnotationImpex.importAnnotation(parse("{}"), TStruct()),
      TStruct(),
    )
    assert(compact(render(actual)) == "{}")
  }

  @Test def testRoundTripEmptyStruct(): Unit = {
    val actual = JSONAnnotationImpex.importAnnotation(
      JSONAnnotationImpex.exportAnnotation(Row(), TStruct()),
      TStruct(),
    )
    assert(actual == Row())
  }

  @Test def testImpexes(): Unit = {

    val g = for {
      t <- arbitrary[Type]
      a <- genNullable(ctx, t)
    } yield (t, a)

    forAll(g) { case (t, a) =>
      JSONAnnotationImpex.importAnnotation(JSONAnnotationImpex.exportAnnotation(a, t), t) == a
    }

    forAll(g) { case (t, a) =>
      val string = compact(JSONAnnotationImpex.exportAnnotation(a, t))
      JSONAnnotationImpex.importAnnotation(parse(string), t) == a
    }
  }

  @Test def testOrdering(): Unit = {
    val intOrd = TInt32.ordering(ctx.stateManager)

    assert(intOrd.compare(-2, -2) == 0)
    assert(intOrd.compare(null, null) == 0)
    assert(intOrd.compare(5, 7) < 0)
    assert(intOrd.compare(5, null) < 0)
    assert(intOrd.compare(null, -2) > 0)

    val g = for {
      t <- arbitrary[Type]
      a <- genNullable(ctx, t)
      b <- genNullable(ctx, t)
    } yield (t, a, b)

    forAll(g) { case (t, a, b) =>
      val ord = t.ordering(ctx.stateManager)
      ord.compare(a, b) == -ord.compare(b, a)
    }
  }
}
