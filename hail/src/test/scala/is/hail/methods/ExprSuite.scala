package is.hail.methods

import is.hail.HailSuite
import is.hail.backend.HailStateManager
import is.hail.check.Prop._
import is.hail.check.Properties
import is.hail.expr._
import is.hail.expr.ir.IRParser
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils.StringEscapeUtils._

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.testng.annotations.Test

class ExprSuite extends HailSuite {

  def sm: HailStateManager = ctx.stateManager

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
      t == parsed
    })
  }

  @Test def testEscaping(): Unit = {
    val p = forAll((s: String) => s == unescapeString(escapeString(s)))

    p.check()
  }

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

    val p = forAll { (s: String) =>
      s == unescapeStringSimple(
        escapeStringSimple(s, '_', _.isLetterOrDigit, _.isLetterOrDigit),
        '_',
      )
    }

    p.check()
  }

  @Test def testImpexes(): Unit = {

    val g = for {
      t <- Type.genArb
      a <- t.genValue(sm)
    } yield (t, a)

    object Spec extends Properties("ImpEx") {
      property("json") = forAll(g) { case (t, a) =>
        JSONAnnotationImpex.importAnnotation(JSONAnnotationImpex.exportAnnotation(a, t), t) == a
      }

      property("json-text") = forAll(g) { case (t, a) =>
        val string = compact(JSONAnnotationImpex.exportAnnotation(a, t))
        JSONAnnotationImpex.importAnnotation(parse(string), t) == a
      }
    }

    Spec.check()
  }

  @Test def testOrdering(): Unit = {
    val intOrd = TInt32.ordering(ctx.stateManager)

    assert(intOrd.compare(-2, -2) == 0)
    assert(intOrd.compare(null, null) == 0)
    assert(intOrd.compare(5, 7) < 0)
    assert(intOrd.compare(5, null) < 0)
    assert(intOrd.compare(null, -2) > 0)

    val g = for {
      t <- Type.genArb
      a <- t.genValue(sm)
      b <- t.genValue(sm)
    } yield (t, a, b)

    val p = forAll(g) { case (t, a, b) =>
      val ord = t.ordering(ctx.stateManager)
      ord.compare(a, b) == -ord.compare(b, a)
    }
    p.check()
  }
}
