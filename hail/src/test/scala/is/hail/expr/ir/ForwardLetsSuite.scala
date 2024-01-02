package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.TestUtils._
import is.hail.expr.Nat
import is.hail.expr.ir.DeprecatedIRBuilder.{applyAggOp, let, _}
import is.hail.types.virtual._
import is.hail.utils._
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.Matchers.{be, convertToAnyShouldWrapper}
import org.testng.annotations.{BeforeMethod, DataProvider, Test}

class ForwardLetsSuite extends HailSuite {

  @BeforeMethod
  def resetUidCounter(): Unit = {
    is.hail.expr.ir.uidCounter = 0
  }

  @DataProvider(name = "nonForwardingOps")
  def nonForwardingOps(): Array[Array[IR]] = {
    val a = ToArray(StreamRange(I32(0), I32(10), I32(1)))
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    Array(
      ToArray(StreamMap(ToStream(a), "y", ApplyBinaryPrimOp(Add(), x, y))),
      ToArray(StreamFilter(ToStream(a), "y", ApplyComparisonOp(LT(TInt32), x, y))),
      ToArray(StreamFlatMap(ToStream(a), "y", StreamRange(x, y, I32(1)))),
      StreamFold(ToStream(a), I32(0), "acc", "y", ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), Ref("acc", TInt32))),
      StreamFold2(ToStream(a), FastSeq(("acc", I32(0))), "y", FastSeq(x + y + Ref("acc", TInt32)), Ref("acc", TInt32)),
      ToArray(StreamScan(ToStream(a), I32(0), "acc", "y", ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), Ref("acc", TInt32)))),
      MakeStruct(FastSeq("a" -> ApplyBinaryPrimOp(Add(), x, I32(1)), "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)))),
      MakeTuple.ordered(FastSeq(ApplyBinaryPrimOp(Add(), x, I32(1)), ApplyBinaryPrimOp(Add(), x, I32(2)))),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, x), I32(1)),
      StreamAgg(ToStream(a), "y", ApplyAggOp(Sum())(x + y))
    ).map(ir => Array[IR](Let(FastSeq("x" -> (In(0, TInt32) + In(0, TInt32))), ir)))
  }

  @DataProvider(name = "nonForwardingNonEvalOps")
  def nonForwardingNonEvalOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    Array(
      NDArrayMap(In(1, TNDArray(TInt32, Nat(1))), "y", x + y),
      NDArrayMap2(In(1, TNDArray(TInt32, Nat(1))), In(2, TNDArray(TInt32, Nat(1))), "y", "z", x + y + Ref("z", TInt32), ErrorIDs.NO_ERROR),
      TailLoop("f", FastSeq("y" -> I32(0)), TInt32, If(y < x, Recur("f", FastSeq[IR](y - I32(1)), TInt32), x))
    ).map(ir => Array[IR](Let(FastSeq("x" -> (In(0, TInt32) + In(0, TInt32))), ir)))
  }

  def aggMin(value: IR): ApplyAggOp = ApplyAggOp(FastSeq(), FastSeq(value), AggSignature(Min(), FastSeq(), FastSeq(value.typ)))

  @DataProvider(name = "nonForwardingAggOps")
  def nonForwardingAggOps(): Array[Array[IR]] = {
    val a = StreamRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    Array(
      AggArrayPerElement(ToArray(a), "y", "_", aggMin(x + y), None, false),
      AggExplode(a, "y", aggMin(y + x), false)
    ).map(ir => Array[IR](AggLet("x", In(0, TInt32) + In(0, TInt32), ir, false)))
  }

  @DataProvider(name = "forwardingOps")
  def forwardingOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32)
    Array(
      MakeStruct(FastSeq("a" -> I32(1), "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)))),
      MakeTuple.ordered(FastSeq(I32(1), ApplyBinaryPrimOp(Add(), x, I32(2)))),
      If(True(), x, I32(0)),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), I32(2), x), I32(1)),
      ApplyUnaryPrimOp(Negate, x),
      ToArray(StreamMap(StreamRange(I32(0), x, I32(1)), "foo", Ref("foo", TInt32))),
      ToArray(StreamFilter(StreamRange(I32(0), x, I32(1)), "foo", Ref("foo", TInt32) <= I32(0)))
    ).map(ir => Array[IR](Let(FastSeq("x" -> (In(0, TInt32) + In(0, TInt32))), ir)))
  }

  @DataProvider(name = "forwardingAggOps")
  def forwardingAggOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32)
    val other = Ref("other", TInt32)
    Array(
      AggFilter(x.ceq(I32(0)), aggMin(other), false),
      aggMin(x + other)
    ).map(ir => Array[IR](AggLet("x", In(0, TInt32) + In(0, TInt32), ir, false)))
  }

  @Test def assertDataProvidersWork() {
    nonForwardingOps()
    forwardingOps()
    nonForwardingAggOps()
    forwardingAggOps()
  }

  @Test(dataProvider = "nonForwardingOps")
  def testNonForwardingOps(ir: IR): Unit = {
    val after = ForwardLets(ctx)(ir)
    val normalizedBefore = (new NormalizeNames(_.toString))(ctx, ir)
    val normalizedAfter = (new NormalizeNames(_.toString))(ctx, after)
    assert(normalizedBefore == normalizedAfter)
  }

  @Test(dataProvider = "nonForwardingNonEvalOps")
  def testNonForwardingNonEvalOps(ir: IR): Unit = {
    val after = ForwardLets(ctx)(ir)
    assert(after.isInstanceOf[Let])
  }

  @Test(dataProvider = "nonForwardingAggOps")
  def testNonForwardingAggOps(ir: IR): Unit = {
    val after = ForwardLets(ctx)(ir)
    assert(after.isInstanceOf[AggLet])
  }

  @Test(dataProvider = "forwardingOps")
  def testForwardingOps(ir: IR): Unit = {
    val after = ForwardLets(ctx)(ir)
    assert(!after.isInstanceOf[Let])
    assertEvalSame(ir, args = Array(5 -> TInt32))
  }

  @Test(dataProvider = "forwardingAggOps")
  def testForwardingAggOps(ir: IR): Unit = {
    val after = ForwardLets(ctx)(ir)
    assert(!after.isInstanceOf[AggLet])
  }

  @DataProvider(name = "TrivialIRCases")
  def trivalIRCases: Array[Array[Any]] = {
    val pi = Math.atan(1) * 4

    Array(
      Array(
        Let(FastSeq("x" -> I32(0)), I32(2)),
        I32(2),
        """"x" is unused."""
      ),
      Array(
        Let(FastSeq("x" -> I32(0)), Ref("x", TInt32)),
        I32(0),
        """"x" is constant and is used once."""
      ),
      Array(
        Let(FastSeq("x" -> I32(2)), Ref("x", TInt32) * Ref("x", TInt32)),
        I32(2) * I32(2),
        """"x" is a primitive constant (ForwardLets does not evaluate)."""
      ),
      Array(
        bindIRs(I32(2), F64(pi), Ref("r", TFloat64)) { case Seq(two, pi, r) =>
          ApplyBinaryPrimOp(Multiply(),
            ApplyBinaryPrimOp(Multiply(), Cast(two, TFloat64), pi),
            r
          )
        },
        ApplyBinaryPrimOp(Multiply(),
          ApplyBinaryPrimOp(Multiply(), Cast(I32(2), TFloat64), F64(pi)),
          Ref("r", TFloat64)
        ),
        """Forward constant primitive values and simple use ref."""
      ),
      Array(
        Let(
          FastSeq(
            iruid(0) -> I32(2),
            iruid(1) -> Cast(Ref(iruid(0), TInt32), TFloat64),
            iruid(2) -> ApplyBinaryPrimOp(FloatingPointDivide(), Ref(iruid(1), TFloat64), F64(2)),
            iruid(3) -> F64(pi),
            iruid(4) -> ApplyBinaryPrimOp(Multiply(), Ref(iruid(3), TFloat64), Ref(iruid(1), TFloat64)),
            iruid(5) -> ApplyBinaryPrimOp(Multiply(), Ref(iruid(2), TFloat64), Ref(iruid(2), TFloat64)),
            iruid(6) -> ApplyBinaryPrimOp(Multiply(), Ref(iruid(3), TFloat64), Ref(iruid(5), TFloat64))
          ),
          MakeStruct(FastSeq(
            "radius" -> Ref(iruid(2), TFloat64),
            "circumference" -> Ref(iruid(4), TFloat64),
            "area" -> Ref(iruid(6), TFloat64),
          ))
        ),
        Let(FastSeq(
          iruid(1) -> Cast(I32(2), TFloat64),
          iruid(2) -> ApplyBinaryPrimOp(FloatingPointDivide(), Ref(iruid(1), TFloat64), F64(2)),
        ),
          MakeStruct(FastSeq(
            "radius" -> Ref(iruid(2), TFloat64),
            "circumference" -> ApplyBinaryPrimOp(Multiply(), F64(pi), Ref(iruid(1), TFloat64)),
            "area" -> ApplyBinaryPrimOp(Multiply(), F64(pi),
              ApplyBinaryPrimOp(Multiply(), Ref(iruid(2), TFloat64), Ref(iruid(2), TFloat64))
            )
          ))
        ),
        "Cascading Let-bindings are forwarded"
      )
    )
  }

  @Test(dataProvider = "TrivialIRCases")
  def testTrivialCases(input: IR, expected: IR, reason: String): Unit =
    ForwardLets(ctx)(input) should be(expected) withClue reason

  @Test def testAggregators(): Unit = {
    val aggEnv = Env[Type]("row" -> TStruct("idx" -> TInt32))
    val ir0 = applyAggOp(Sum(), seqOpArgs = FastSeq(let(x = 'row('idx) - 1) {
      'x.toD
    }))
      .apply(aggEnv)

    TypeCheck(ctx, ForwardLets(ctx)(ir0), BindingEnv(Env.empty, agg = Some(aggEnv)))
  }

  @Test def testNestedBindingOverwrites(): Unit = {
    val env = Env[Type]("x" -> TInt32)
    val ir = let(y = 'x.toD, x = 'x.toD) {
      'x + 'x + 'y
    }(env)

    TypeCheck(ctx, ir, BindingEnv(env))
    TypeCheck(ctx, ForwardLets(ctx)(ir), BindingEnv(env))
  }

  @Test def testLetsDoNotForwardInsideArrayAggWithNoOps(): Unit = {
    val x = Let(
      FastSeq(
        "x" -> StreamAgg(ToStream(In(0, TArray(TInt32))), "foo", Ref("y", TInt32))
      ),
      StreamAgg(
        ToStream(In(1, TArray(TInt32))),
        "bar",
        Ref("y", TInt32) + Ref("x", TInt32)
      )
    )

    TypeCheck(ctx, x, BindingEnv(Env("y" -> TInt32)))
    TypeCheck(ctx, ForwardLets(ctx)(x), BindingEnv(Env("y" -> TInt32)))
  }
}
