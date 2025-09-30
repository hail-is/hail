package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.expr.ir.defs.{ApplyBinaryPrimOp, I64, MakeStruct, TableCount, TableGetGlobals}
import is.hail.expr.ir.lowering.ExecuteRelational
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LiftLiteralsSuite extends HailSuite {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly

  @Test def testNestedGlobalsRewrite(): Unit = {
    val tab =
      TableLiteral(ExecuteRelational(ctx, TableRange(10, 1)).asTableValue(ctx), theHailClassLoader)
    val ir = TableGetGlobals(
      TableMapGlobals(
        tab,
        bindIR(I64(1)) { global =>
          MakeStruct(
            FastSeq(
              "x" -> ApplyBinaryPrimOp(
                Add(),
                TableCount(tab),
                global,
              )
            )
          )
        },
      )
    )

    assertEvalsTo(ir, Row(11L))
  }
}
