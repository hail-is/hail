package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils.IRCall
import is.hail.types.virtual.{TArray, TBoolean, TCall, TInt32}
import is.hail.variant._

import org.testng.annotations.{DataProvider, Test}

class CallFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = {
    assert(true)
    Array(
      Array(Call0()),
      Array(Call1(0, false)),
      Array(Call1(1, true)),
      Array(Call2(1, 0, true)),
      Array(Call2(0, 1, false)),
      Array(CallN(Array(1, 1), false)),
      Array(Call.parse("0|1")),
    )
  }

  @DataProvider(name = "diploid")
  def uphasedDiploidData(): Array[Array[Any]] = {
    assert(true)
    Array(
      Array(Call2(0, 0, false)),
      Array(Call2(1, 0, false)),
      Array(Call2(0, 1, false)),
      Array(Call2(3, 1, false)),
      Array(Call2(3, 3, false)),
    )
  }

  @DataProvider(name = "basicWithIndex")
  def basicDataWIndex(): Array[Array[Any]] = {
    assert(true)
    Array(
      Array(Call1(0, false), 0),
      Array(Call1(1, true), 0),
      Array(Call2(1, 0, true), 0),
      Array(Call2(1, 0, true), 1),
      Array(Call2(0, 1, false), 0),
      Array(Call2(0, 1, false), 1),
      Array(CallN(Array(1, 1), false), 0),
      Array(CallN(Array(1, 1), false), 1),
      Array(Call.parse("0|1"), 0),
      Array(Call.parse("0|1"), 1),
    )
  }

  @Test def constructors() {
    assertEvalsTo(invoke("Call", TCall, False()), Call0())
    assertEvalsTo(invoke("Call", TCall, I32(0), True()), Call1(0, true))
    assertEvalsTo(invoke("Call", TCall, I32(1), False()), Call1(1, false))
    assertEvalsTo(invoke("Call", TCall, I32(0), I32(0), False()), Call2(0, 0, false))
    assertEvalsTo(
      invoke("Call", TCall, TestUtils.IRArray(0, 1), False()),
      CallN(Array(0, 1), false),
    )
    assertEvalsTo(invoke("Call", TCall, Str("0|1")), Call2(0, 1, true))
  }

  @Test(dataProvider = "basic")
  def isPhased(c: Call) {
    assertEvalsTo(invoke("isPhased", TBoolean, IRCall(c)), Option(c).map(Call.isPhased).orNull)
  }

  @Test(dataProvider = "basic")
  def isHomRef(c: Call) {
    assertEvalsTo(invoke("isHomRef", TBoolean, IRCall(c)), Option(c).map(Call.isHomRef).orNull)
  }

  @Test(dataProvider = "basic")
  def isHet(c: Call) {
    assertEvalsTo(invoke("isHet", TBoolean, IRCall(c)), Option(c).map(Call.isHet).orNull)
  }

  @Test(dataProvider = "basic")
  def isHomVar(c: Call) {
    assertEvalsTo(invoke("isHomVar", TBoolean, IRCall(c)), Option(c).map(Call.isHomVar).orNull)
  }

  @Test(dataProvider = "basic")
  def isNonRef(c: Call) {
    assertEvalsTo(invoke("isNonRef", TBoolean, IRCall(c)), Option(c).map(Call.isNonRef).orNull)
  }

  @Test(dataProvider = "basic")
  def isHetNonRef(c: Call) {
    assertEvalsTo(
      invoke("isHetNonRef", TBoolean, IRCall(c)),
      Option(c).map(Call.isHetNonRef).orNull,
    )
  }

  @Test(dataProvider = "basic")
  def isHetRef(c: Call) {
    assertEvalsTo(invoke("isHetRef", TBoolean, IRCall(c)), Option(c).map(Call.isHetRef).orNull)
  }

  @Test(dataProvider = "basic")
  def nNonRefAlleles(c: Call) {
    assertEvalsTo(
      invoke("nNonRefAlleles", TInt32, IRCall(c)),
      Option(c).map(Call.nNonRefAlleles).orNull,
    )
  }

  @Test(dataProvider = "basicWithIndex")
  def alleleByIndex(c: Call, idx: Int) {
    assertEvalsTo(
      invoke("index", TInt32, IRCall(c), I32(idx)),
      Option(c).map(c => Call.alleleByIndex(c, idx)).orNull,
    )
  }

  @Test(dataProvider = "basicWithIndex")
  def downcode(c: Call, idx: Int) {
    assertEvalsTo(
      invoke("downcode", TCall, IRCall(c), I32(idx)),
      Option(c).map(c => Call.downcode(c, idx)).orNull,
    )
  }

  @Test(dataProvider = "diploid")
  def unphasedDiploidGtIndex(c: Call) {
    assertEvalsTo(
      invoke("unphasedDiploidGtIndex", TInt32, IRCall(c)),
      Option(c).map(c => Call.unphasedDiploidGtIndex(c)).orNull,
    )
  }

  @Test(dataProvider = "basic")
  def oneHotAlleles(c: Call) {
    assertEvalsTo(
      invoke("oneHotAlleles", TArray(TInt32), IRCall(c), I32(2)),
      Option(c).map(c => Call.oneHotAlleles(c, 2)).orNull,
    )
  }
}
