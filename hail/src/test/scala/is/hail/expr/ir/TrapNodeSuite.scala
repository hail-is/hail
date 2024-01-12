package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class TrapNodeSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def testTrapNode() {
    assertEvalsTo(Trap(ArrayRef(Literal(TArray(TInt32), FastSeq(0, 1, 2)), I32(1))), Row(null, 1))
    val res = eval(Trap(ArrayRef(Literal(TArray(TInt32), FastSeq(0, 1, 2)), I32(-1))))
    res match {
      case Row(Row(msg: String, id: Int), null) =>
        assert(id == -1)
        assert(msg.contains("array index out of bounds"))
    }

    assertEvalsTo(Trap(Die(Str("foo bar"), TInt32, 5)), Row(Row("foo bar", 5), null))
  }

  @Test def testTrapNodeInLargerContext() {
    def resultByIdx(idx: Int): IR =
      bindIR(Trap(ArrayRef(Literal(TArray(TInt32), FastSeq(100, 200, 300)), I32(idx)))) { value =>
        If(IsNA(GetTupleElement(value, 0)), GetTupleElement(value, 1), I32(-1))
      }

    assertEvalsTo(resultByIdx(-100), -1)
    assertEvalsTo(resultByIdx(2), 300)
    assertEvalsTo(resultByIdx(4), -1)
  }
}
