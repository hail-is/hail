package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils.{IRArray, IRCall}
import is.hail.expr.ir.defs.{False, I32, Str, True}
import is.hail.types.virtual.{TArray, TBoolean, TCall, TInt32}
import is.hail.variant._

import org.junit.jupiter.api.Test

class CallFunctionsSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  def basicData() = ArraySeq(
    Call0(),
    Call1(0, false),
    Call1(1, true),
    Call2(1, 0, true),
    Call2(0, 1, false),
    CallN(ArraySeq(1, 1), false),
    Call.parse("0|1"),
  )

  def diploidData() = ArraySeq(
    Call2(0, 0, false),
    Call2(1, 0, false),
    Call2(0, 1, false),
    Call2(3, 1, false),
    Call2(3, 3, false),
  )

  def basicWithIndexData() = ArraySeq[(Call, Int)](
    (Call1(0, false), 0),
    (Call1(1, true), 0),
    (Call2(1, 0, true), 0),
    (Call2(1, 0, true), 1),
    (Call2(0, 1, false), 0),
    (Call2(0, 1, false), 1),
    (CallN(ArraySeq(1, 1), false), 0),
    (CallN(ArraySeq(1, 1), false), 1),
    (Call.parse("0|1"), 0),
    (Call.parse("0|1"), 1),
  )

  @Test def constructors(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("Call", TCall, False()), Call0())
    assertEvalsTo(invoke("Call", TCall, I32(0), True()), Call1(0, true))
    assertEvalsTo(invoke("Call", TCall, I32(1), False()), Call1(1, false))
    assertEvalsTo(invoke("Call", TCall, I32(0), I32(0), False()), Call2(0, 0, false))
    assertEvalsTo(
      invoke("Call", TCall, IRArray(0, 1), False()),
      CallN(ArraySeq(0, 1), false),
    )
    assertEvalsTo(invoke("Call", TCall, Str("0|1")), Call2(0, 1, true))
  }

  @ParameterizedTest("basicData")
  def isPhased(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isPhased", TBoolean, IRCall(c)), Option(c).map(Call.isPhased).orNull)

  @ParameterizedTest("basicData")
  def isHomRef(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isHomRef", TBoolean, IRCall(c)), Option(c).map(Call.isHomRef).orNull)

  @ParameterizedTest("basicData")
  def isHet(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isHet", TBoolean, IRCall(c)), Option(c).map(Call.isHet).orNull)

  @ParameterizedTest("basicData")
  def isHomVar(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isHomVar", TBoolean, IRCall(c)), Option(c).map(Call.isHomVar).orNull)

  @ParameterizedTest("basicData")
  def isNonRef(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isNonRef", TBoolean, IRCall(c)), Option(c).map(Call.isNonRef).orNull)

  @ParameterizedTest("basicData")
  def isHetNonRef(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("isHetNonRef", TBoolean, IRCall(c)),
      Option(c).map(Call.isHetNonRef).orNull,
    )

  @ParameterizedTest("basicData")
  def isHetRef(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isHetRef", TBoolean, IRCall(c)), Option(c).map(Call.isHetRef).orNull)

  @ParameterizedTest("basicData")
  def nNonRefAlleles(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("nNonRefAlleles", TInt32, IRCall(c)),
      Option(c).map(Call.nNonRefAlleles).orNull,
    )

  @ParameterizedTest("basicWithIndexData")
  def alleleByIndex(c: Call, idx: Int)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("index", TInt32, IRCall(c), I32(idx)),
      Option(c).map(c => Call.alleleByIndex(c, idx)).orNull,
    )

  @ParameterizedTest("basicWithIndexData")
  def downcode(c: Call, idx: Int)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("downcode", TCall, IRCall(c), I32(idx)),
      Option(c).map(c => Call.downcode(c, idx)).orNull,
    )

  @ParameterizedTest("diploidData")
  def unphasedDiploidGtIndex(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("unphasedDiploidGtIndex", TInt32, IRCall(c)),
      Option(c).map(c => Call.unphasedDiploidGtIndex(c)).orNull,
    )

  @ParameterizedTest("basicData")
  def oneHotAlleles(c: Call)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("oneHotAlleles", TArray(TInt32), IRCall(c), I32(2)),
      Option(c).map(c => Call.oneHotAlleles(c, 2)).orNull,
    )
}
