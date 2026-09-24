package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{ApplyBinaryPrimOp, I64, MakeStruct, TableCount, TableGetGlobals}
import is.hail.expr.ir.lowering.ExecuteRelational

import org.junit.jupiter.api.Test

class LiftLiteralsSuite {
  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.interpretOnly

  @Test def testNestedGlobalsRewrite(implicit ctx: ExecuteContext): Unit = {
    val tab =
      TableLiteral(
        ExecuteRelational(ctx, TableRange(10, 1)).asTableValue(ctx),
        ctx.theHailClassLoader,
      )
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

    assertEvalsTo(ir, RowSeq(11L))
  }
}
