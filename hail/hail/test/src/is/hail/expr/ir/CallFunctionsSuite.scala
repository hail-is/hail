package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils.{IRArray, IRCall}
import is.hail.expr.ir.defs.{False, I32, Str, True}
import is.hail.types.virtual.{TArray, TBoolean, TCall, TInt32}
import is.hail.variant._

class CallFunctionsSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  val basicData: Array[Call] = Array(
    Call0(),
    Call1(0, false),
    Call1(1, true),
    Call2(1, 0, true),
    Call2(0, 1, false),
    CallN(ArraySeq(1, 1), false),
    Call.parse("0|1"),
  )

  val diploidData: Array[Call] = Array(
    Call2(0, 0, false),
    Call2(1, 0, false),
    Call2(0, 1, false),
    Call2(3, 1, false),
    Call2(3, 3, false),
  )

  val basicWithIndexData: Array[(Call, Int)] = Array(
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

  test("constructors") {
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

  object checkIsPhased extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isPhased") {
      assertEvalsTo(invoke("isPhased", TBoolean, IRCall(c)), Option(c).map(Call.isPhased).orNull)
    }
  }

  basicData.foreach(checkIsPhased(_))

  object checkIsHomRef extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isHomRef") {
      assertEvalsTo(invoke("isHomRef", TBoolean, IRCall(c)), Option(c).map(Call.isHomRef).orNull)
    }
  }

  basicData.foreach(checkIsHomRef(_))

  object checkIsHet extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isHet") {
      assertEvalsTo(invoke("isHet", TBoolean, IRCall(c)), Option(c).map(Call.isHet).orNull)
    }
  }

  basicData.foreach(checkIsHet(_))

  object checkIsHomVar extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isHomVar") {
      assertEvalsTo(invoke("isHomVar", TBoolean, IRCall(c)), Option(c).map(Call.isHomVar).orNull)
    }
  }

  basicData.foreach(checkIsHomVar(_))

  object checkIsNonRef extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isNonRef") {
      assertEvalsTo(invoke("isNonRef", TBoolean, IRCall(c)), Option(c).map(Call.isNonRef).orNull)
    }
  }

  basicData.foreach(checkIsNonRef(_))

  object checkIsHetNonRef extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isHetNonRef") {
      assertEvalsTo(
        invoke("isHetNonRef", TBoolean, IRCall(c)),
        Option(c).map(Call.isHetNonRef).orNull,
      )
    }
  }

  basicData.foreach(checkIsHetNonRef(_))

  object checkIsHetRef extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("isHetRef") {
      assertEvalsTo(invoke("isHetRef", TBoolean, IRCall(c)), Option(c).map(Call.isHetRef).orNull)
    }
  }

  basicData.foreach(checkIsHetRef(_))

  object checkNNonRefAlleles extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("nNonRefAlleles") {
      assertEvalsTo(
        invoke("nNonRefAlleles", TInt32, IRCall(c)),
        Option(c).map(Call.nNonRefAlleles).orNull,
      )
    }
  }

  basicData.foreach(checkNNonRefAlleles(_))

  object checkAlleleByIndex extends TestCases {
    def apply(c: Call, idx: Int)(implicit loc: munit.Location): Unit = test("alleleByIndex") {
      assertEvalsTo(
        invoke("index", TInt32, IRCall(c), I32(idx)),
        Option(c).map(c => Call.alleleByIndex(c, idx)).orNull,
      )
    }
  }

  basicWithIndexData.foreach { case (c, idx) => checkAlleleByIndex(c, idx) }

  object checkDowncode extends TestCases {
    def apply(c: Call, idx: Int)(implicit loc: munit.Location): Unit = test("downcode") {
      assertEvalsTo(
        invoke("downcode", TCall, IRCall(c), I32(idx)),
        Option(c).map(c => Call.downcode(c, idx)).orNull,
      )
    }
  }

  basicWithIndexData.foreach { case (c, idx) => checkDowncode(c, idx) }

  object checkUnphasedDiploidGtIndex extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("unphasedDiploidGtIndex") {
      assertEvalsTo(
        invoke("unphasedDiploidGtIndex", TInt32, IRCall(c)),
        Option(c).map(c => Call.unphasedDiploidGtIndex(c)).orNull,
      )
    }
  }

  diploidData.foreach(checkUnphasedDiploidGtIndex(_))

  object checkOneHotAlleles extends TestCases {
    def apply(c: Call)(implicit loc: munit.Location): Unit = test("oneHotAlleles") {
      assertEvalsTo(
        invoke("oneHotAlleles", TArray(TInt32), IRCall(c), I32(2)),
        Option(c).map(c => Call.oneHotAlleles(c, 2)).orNull,
      )
    }
  }

  basicData.foreach(checkOneHotAlleles(_))
}
