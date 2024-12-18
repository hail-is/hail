package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LiftLiteralsSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.interpretOnly

  @Test def testNestedGlobalsRewrite(): Unit = {
    val tab =
      TableLiteral(TableRange(10, 1).analyzeAndExecute(ctx).asTableValue(ctx), theHailClassLoader)
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
