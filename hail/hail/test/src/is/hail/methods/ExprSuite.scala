package is.hail.methods

import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.expr._
import is.hail.expr.ir.IRParser
import is.hail.scalacheck._
import is.hail.types.virtual._
import is.hail.utils.StringEscapeUtils._

import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.junit.jupiter.api.Test
import org.scalacheck.Arbitrary._
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class ExprSuite {

  def sm(implicit ctx: ExecuteContext): HailStateManager = ctx.stateManager

  @Test def testTypePretty(): Unit = {
    // for arbType

    val sb = new StringBuilder
    check(forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, 0, compact = true)
      val res = sb.result()
      val parsed = IRParser.parseType(res)
      t == parsed
    })

    check(forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, 0, compact = false)
      val res = sb.result()
      val parsed = IRParser.parseType(res)
      t == parsed
    })

    check(forAll { (t: Type) =>
      val s = t.parsableString()
      val parsed = IRParser.parseType(s)
      assert(t == parsed)
    })
  }

  @Test def testEscaping(): Unit =
    check(forAll((s: String) => assert(s == unescapeString(escapeString(s)))))

  @Test def testEscapingSimple(): Unit = {
    // a == 0x61, _ = 0x5f
    assertEq(escapeStringSimple("abc", '_', _ => false), "abc")
    assertEq(escapeStringSimple("abc", '_', _ == 'a'), "_61bc")
    assertEq(escapeStringSimple("abc_", '_', _ => false), "abc_5f")
    assertEq(unescapeStringSimple("abc", '_'), "abc")
    assertEq(unescapeStringSimple("abc_5f", '_'), "abc_")
    assertEq(unescapeStringSimple("_61bc", '_'), "abc")
    assertEq(unescapeStringSimple("_u0061bc", '_'), "abc")
    assertEq(escapeStringSimple("my name is 名谦", '_', _ => false), "my name is _u540d_u8c26")
    assertEq(unescapeStringSimple("my name is _u540d_u8c26", '_'), "my name is 名谦")

    check(forAll(Gen.asciiPrintableStr) { (s: String) =>
      assert(s == unescapeStringSimple(
        escapeStringSimple(s, '_', _.isLetterOrDigit, _.isLetterOrDigit),
        '_',
      ))
    })
  }

  @Test def testImportEmptyJSONObjectAsStruct(): Unit =
    assertEq(JSONAnnotationImpex.importAnnotation(parse("{}"), TStruct()), RowSeq())

  @Test def testExportEmptyJSONObjectAsStruct(): Unit =
    assertEq(compact(render(JSONAnnotationImpex.exportAnnotation(RowSeq(), TStruct()))), "{}")

  @Test def testRoundTripEmptyJSONObject(): Unit = {
    val actual = JSONAnnotationImpex.exportAnnotation(
      JSONAnnotationImpex.importAnnotation(parse("{}"), TStruct()),
      TStruct(),
    )
    assertEq(compact(render(actual)), "{}")
  }

  @Test def testRoundTripEmptyStruct(): Unit = {
    val actual = JSONAnnotationImpex.importAnnotation(
      JSONAnnotationImpex.exportAnnotation(RowSeq(), TStruct()),
      TStruct(),
    )
    assertEq(actual, RowSeq())
  }

  @Test def testImpexes(implicit ctx: ExecuteContext): Unit = {

    val g = for {
      t <- arbitrary[Type]
      a <- genNullable(ctx, t)
    } yield (t, a)

    check(forAll(g) { case (t, a) =>
      assert(JSONAnnotationImpex.importAnnotation(
        JSONAnnotationImpex.exportAnnotation(a, t),
        t,
      ) == a)
    })

    check(forAll(g) { case (t, a) =>
      val string = compact(JSONAnnotationImpex.exportAnnotation(a, t))
      assert(JSONAnnotationImpex.importAnnotation(parse(string), t) == a)
    })
  }

  @Test def testOrdering(implicit ctx: ExecuteContext): Unit = {
    val intOrd = TInt32.ordering(ctx.stateManager)

    assertEq(intOrd.compare(-2, -2), 0)
    assertEq(intOrd.compare(null, null), 0)
    assert(intOrd.compare(5, 7) < 0)
    assert(intOrd.compare(5, null) < 0)
    assert(intOrd.compare(null, -2) > 0)

    val g = for {
      t <- arbitrary[Type]
      a <- genNullable(ctx, t)
      b <- genNullable(ctx, t)
    } yield (t, a, b)

    check(forAll(g) { case (t, a, b) =>
      val ord = t.ordering(ctx.stateManager)
      assertEq(ord.compare(a, b), -ord.compare(b, a))
    })
  }
}
