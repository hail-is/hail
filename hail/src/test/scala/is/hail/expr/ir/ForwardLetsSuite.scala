package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.TestUtils._
import is.hail.expr.Nat
import is.hail.expr.ir.DeprecatedIRBuilder.{applyAggOp, let, _}
import is.hail.types.virtual._
import is.hail.utils._
import org.testng.annotations.{DataProvider, Test}

class ForwardLetsSuite extends HailSuite {
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
    ).map(ir => Array[IR](Let("x", In(0, TInt32) + In(0, TInt32), ir)))
  }

  @DataProvider(name = "nonForwardingNonEvalOps")
  def nonForwardingNonEvalOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32)
    val y = Ref("y", TInt32)
    Array(
      NDArrayMap(In(1, TNDArray(TInt32, Nat(1))), "y", x + y),
      NDArrayMap2(In(1, TNDArray(TInt32, Nat(1))), In(2, TNDArray(TInt32, Nat(1))), "y", "z", x + y + Ref("z", TInt32), ErrorIDs.NO_ERROR),
      TailLoop("f", FastSeq("y" -> I32(0)), If(y < x, Recur("f", FastSeq[IR](y - I32(1)), TInt32), x))
    ).map(ir => Array[IR](Let("x", In(0, TInt32) + In(0, TInt32), ir)))
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
    ).map(ir => Array[IR](Let("x", In(0, TInt32) + In(0, TInt32), ir)))
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

  @Test def testLetNoMention(): Unit = {
    val ir = Let("x", I32(1), I32(2))
    assert(ForwardLets[IR](ctx)(ir) == I32(2))
  }

  @Test def testLetRefRewrite(): Unit = {
    val ir = Let("x", I32(1), Ref("x", TInt32))
    assert(ForwardLets[IR](ctx)(ir) == I32(1))
  }

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
      "x",
      StreamAgg(
        ToStream(In(0, TArray(TInt32))),
        "foo",
        Ref(
          "y", TInt32)),
      StreamAgg(ToStream(In(1, TArray(TInt32))),
        "bar",
        Ref("y", TInt32) + Ref("x", TInt32
        )))

    TypeCheck(ctx, x, BindingEnv(Env("y" -> TInt32)))
    TypeCheck(ctx, ForwardLets(ctx)(x), BindingEnv(Env("y" -> TInt32)))
  }
}
