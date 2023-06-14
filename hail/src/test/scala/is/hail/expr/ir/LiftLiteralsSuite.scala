package is.hail.expr.ir

import cats.syntax.all._
import is.hail.expr.ir.lowering.Lower.monadLowerInstanceForLower
import is.hail.expr.ir.lowering.{Lower, LoweringState}
import is.hail.types.virtual.TInt64
import is.hail.utils.FastSeq
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LiftLiteralsSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.interpretOnly

  @Test def testNestedGlobalsRewrite() {
    val tab = (for {
        tv <- TableRange(10, 1).analyzeAndExecute[Lower] >>= (_.asTableValue);
        lit <- TableLiteral[Lower](tv)
      } yield lit).runA(ctx, LoweringState())

    val ir = TableGetGlobals(
      TableMapGlobals(
        tab,
        Let(
          "global",
          I64(1),
          MakeStruct(
            FastSeq(
              "x" -> ApplyBinaryPrimOp(
                Add(),
                TableCount(tab),
                Ref("global", TInt64)))))))

    assertEvalsTo(ir, Row(11L))
  }
}
