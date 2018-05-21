package is.hail.expr

import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils.FastSeq
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class ASTToIRSuite extends TestNGSuite {
  private def toIR[T](s: String): Option[IR] = {
    val ast = Parser.parseToAST(s, EvalContext(Map(
      "aggregable" -> (0, TAggregable(TInt64(),
        Map("agg" -> (0, TInt64()),
          "something" -> (1, TInt64())))))))

    ast.toIROpt(Some(("aggregable", "agg"))).map(_.unwrap)
  }

  @Test
  def constants() {
    for {(in, out) <- Array(
      "3" -> I32(3),
      Int.MaxValue.toString -> I32(Int.MaxValue),
      "3.0" -> F64(3.0),
      "true" -> True(),
      "false" -> False(),
      "{}" -> MakeStruct(FastSeq()),
      "{a : 1}" -> MakeStruct(FastSeq(("a", I32(1)))),
      "{a: 1, b: 2}" -> MakeStruct(FastSeq(
        ("a", I32(1)), ("b", I32(2)))),
      "[1, 2]" -> MakeArray(FastSeq(I32(1), I32(2)), TArray(TInt32())),
      "[42.0]" -> MakeArray(FastSeq(F64(42.0)), TArray(TFloat64()))
    )
    } {
      assert(toIR(in).contains(out),
        s"expected '$in' to parse and convert into $out, but got ${ toIR(in) }")
    }
  }

  @Test
  def getField() {
    for {(in, out) <- Array(
      "{a: 1, b: 2}.a" ->
        GetField(
          MakeStruct(FastSeq(
            ("a", I32(1)), ("b", I32(2)))),
          "a"),
      "{a: 1, b: 2}.b" ->
        GetField(
          MakeStruct(FastSeq(
            ("a", I32(1)), ("b", I32(2)))),
          "b")
    )
    } {
      assert(toIR(in).contains(out),
        s"expected '$in' to parse and convert into $out, but got ${ toIR(in) }")
    }
  }

  @Test
  def let() {
    for {(in, out) <- Array(
      "let a = 0 and b = 3 in b" ->
        ir.Let("a", I32(0), ir.Let("b", I32(3), Ref("b", TInt32()))),
      "let a = 0 and b = a in b" ->
        ir.Let("a", I32(0), ir.Let("b", Ref("a", TInt32()), Ref("b", TInt32()))),
      "let i = 7 in i" ->
        ir.Let("i", I32(7), Ref("i", TInt32())),
      "let a = let b = 3 in b in a" ->
        ir.Let("a", ir.Let("b", I32(3), Ref("b", TInt32())), Ref("a", TInt32()))
    )
    } {
      val r = toIR(in)
      assert(r.contains(out),
        s"expected '$in' to parse and convert into $out, but got ${ toIR(in) }")
    }
  }

  @Test
  def primOps() {
    for {(in, out) <- Array(
      "-1" -> ApplyUnaryPrimOp(Negate(), I32(1)),
      "!true" -> ApplyUnaryPrimOp(Bang(), True()),
      "1 / 2" -> ApplyBinaryPrimOp(FloatingPointDivide(), I32(1), I32(2)),
      "1.0 / 2.0" -> ApplyBinaryPrimOp(FloatingPointDivide(), F64(1.0), F64(2.0)),
      "1.0 < 2.0" -> ApplyComparisonOp(LT(TFloat64()), F64(1.0), F64(2.0)),
      "1.0 <= 2.0" -> ApplyComparisonOp(LTEQ(TFloat64()), F64(1.0), F64(2.0)),
      "1.0 >= 2.0" -> ApplyComparisonOp(GTEQ(TFloat64()), F64(1.0), F64(2.0)),
      "1.0 > 2.0" -> ApplyComparisonOp(GT(TFloat64()), F64(1.0), F64(2.0)),
      "1.0 == 2.0" -> ApplyComparisonOp(EQ(TFloat64()), F64(1.0), F64(2.0)),
      "1.0 != 2.0" -> ApplyComparisonOp(NEQ(TFloat64()), F64(1.0), F64(2.0)),
      "0 // 0 + 1" -> ApplyBinaryPrimOp(
        Add(),
        ApplyBinaryPrimOp(RoundToNegInfDivide(), I32(0), I32(0)),
        I32(1)),
      "0 // 0 * 1" -> ApplyBinaryPrimOp(
        Multiply(),
        ApplyBinaryPrimOp(RoundToNegInfDivide(), I32(0), I32(0)),
        I32(1))
    )
    } {
      assert(toIR(in).contains(out),
        s"expected '$in' to parse and convert into $out, but got ${ toIR(in) }")
    }
  }
}
