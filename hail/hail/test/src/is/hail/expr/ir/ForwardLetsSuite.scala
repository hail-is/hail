package is.hail.expr.ir

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.Nat
import is.hail.expr.ir.defs._
import is.hail.types.virtual._

import org.junit.jupiter.api.{BeforeEach, Test}

class ForwardLetsSuite {

  @BeforeEach
  def resetUidCounter(): Unit =
    is.hail.expr.ir.uidCounter = 0

  def testNonForwardingOps() = {
    val a = ToArray(StreamRange(I32(0), I32(10), I32(1)))
    val x = Ref(freshName(), TInt32)
    ArraySeq(
      mapArray(a)(y => ApplyBinaryPrimOp(Add(), x, y)),
      ToArray(filterIR(ToStream(a))(y => ApplyComparisonOp(LT, x, y))),
      ToArray(flatMapIR(ToStream(a))(y => StreamRange(x, y, I32(1)))),
      foldIR(ToStream(a), I32(0)) { (acc, y) =>
        ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), acc)
      },
      fold2IR(ToStream(a), I32(0)) { case (y, Seq(acc)) => x + y + acc } { case Seq(acc) => acc },
      ToArray(streamScanIR(ToStream(a), I32(0))((acc, y) =>
        ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), acc)
      )),
      MakeStruct(FastSeq(
        "a" -> ApplyBinaryPrimOp(Add(), x, I32(1)),
        "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)),
      )),
      MakeTuple.ordered(FastSeq(
        ApplyBinaryPrimOp(Add(), x, I32(1)),
        ApplyBinaryPrimOp(Add(), x, I32(2)),
      )),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, x), I32(1)),
      streamAggIR(ToStream(a))(y => ApplyAggOp(Sum())(x + y)),
    ).map(ir => Let(FastSeq(x.name -> (In(0, TInt32) + In(0, TInt32))), ir))
  }

  def testNonForwardingNonEvalOps() = {
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val z = Ref(freshName(), TInt32)
    val f = freshName()
    ArraySeq(
      NDArrayMap(In(1, TNDArray(TInt32, Nat(1))), y.name, x + y),
      NDArrayMap2(
        In(1, TNDArray(TInt32, Nat(1))),
        In(2, TNDArray(TInt32, Nat(1))),
        y.name,
        z.name,
        x + y + z,
        ErrorIDs.NO_ERROR,
      ),
      TailLoop(
        f,
        FastSeq(y.name -> I32(0)),
        TInt32,
        If(y < x, Recur(f, FastSeq[IR](y - I32(1)), TInt32), x),
      ),
    ).map(ir => Let(FastSeq(x.name -> (In(0, TInt32) + In(0, TInt32))), ir))
  }

  def aggMin(value: IR): ApplyAggOp = ApplyAggOp(Min())(value)

  def testNonForwardingAggOps() = {
    val a = StreamRange(I32(0), I32(10), I32(1))
    val x = Ref(freshName(), TInt32)
    ArraySeq(
      aggArrayPerElement(ToArray(a))((y, _) => aggMin(x + y)),
      aggExplodeIR(a)(y => aggMin(y + x)),
    ).map(ir => AggLet(x.name, In(0, TInt32) + In(0, TInt32), ir, false))
  }

  def testForwardingOps() = {
    val x = Ref(freshName(), TInt32)
    ArraySeq(
      MakeStruct(FastSeq("a" -> I32(1), "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)))),
      MakeTuple.ordered(FastSeq(I32(1), ApplyBinaryPrimOp(Add(), x, I32(2)))),
      If(True(), x, I32(0)),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), I32(2), x), I32(1)),
      ApplyUnaryPrimOp(Negate, x),
      ToArray(mapIR(rangeIR(x))(foo => foo)),
      ToArray(filterIR(rangeIR(x))(foo => foo <= I32(0))),
    ).map(ir => Let(FastSeq(x.name -> (In(0, TInt32) + In(0, TInt32))), ir))
  }

  def testForwardingAggOps() = {
    val x = Ref(freshName(), TInt32)
    val other = Ref(freshName(), TInt32)
    ArraySeq(
      AggFilter(x.ceq(I32(0)), aggMin(other), false),
      aggMin(x + other),
    ).map(ir => AggLet(x.name, In(0, TInt32) + In(0, TInt32), ir, false))
  }

  @Test def assertDataProvidersWork(): Unit = {
    testNonForwardingOps(): Unit
    testForwardingOps(): Unit
    testNonForwardingAggOps(): Unit
    testForwardingAggOps(): Unit
  }

  @Test def testBlock(implicit ctx: ExecuteContext): Unit = {
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val ir = Block(
      FastSeq(Binding(x.name, I32(1), Scope.AGG), Binding(y.name, x, Scope.AGG)),
      ApplyAggOp(Sum())(y),
    )
    val after: IR = ForwardLets(ctx, ir)
    val expected = ApplyAggOp(Sum())(I32(1))
    assertEq(NormalizeNames()(ctx, after), NormalizeNames()(ctx, expected))
  }

  @ParameterizedTest
  def testNonForwardingOps(ir: IR)(implicit ctx: ExecuteContext): Unit = {
    val after = ForwardLets(ctx, ir)
    val normalizedBefore = NormalizeNames()(ctx, ir)
    val normalizedAfter = NormalizeNames()(ctx, after)
    assertEq(normalizedAfter, normalizedBefore)
  }

  @ParameterizedTest
  def testNonForwardingNonEvalOps(ir: IR)(implicit ctx: ExecuteContext): Unit = {
    val after = ForwardLets(ctx, ir)
    assert(after.isInstanceOf[Block])
  }

  @ParameterizedTest
  def testNonForwardingAggOps(ir: IR)(implicit ctx: ExecuteContext): Unit = {
    val after = ForwardLets(ctx, ir)
    assert(after.isInstanceOf[Block])
  }

  @ParameterizedTest
  def testForwardingOps(ir: IR)(implicit ctx: ExecuteContext): Unit = {
    val after = ForwardLets(ctx, ir)
    assert(!after.isInstanceOf[Block])
    assertEvalSame(ir, args = ArraySeq(5 -> TInt32))
  }

  @ParameterizedTest
  def testForwardingAggOps(ir: IR)(implicit ctx: ExecuteContext): Unit = {
    val after = ForwardLets(ctx, ir)
    assert(!after.isInstanceOf[Block])
  }

  def testTrivialCases() = {
    val pi = Math.atan(1) * 4

    val r = Ref(freshName(), TFloat64)
    ArraySeq[(IR, IR, String)](
      (
        bindIR(I32(0))(_ => I32(2)),
        I32(2),
        """"x" is unused.""",
      ),
      (
        bindIR(I32(0))(x => x),
        I32(0),
        """"x" is constant and is used once.""",
      ),
      (
        bindIR(I32(2))(x => x * x),
        I32(2) * I32(2),
        """"x" is a primitive constant (ForwardLets does not evaluate).""",
      ),
      (
        bindIRs(I32(2), F64(pi), r) { case Seq(two, pi, r) =>
          ApplyBinaryPrimOp(Multiply(), ApplyBinaryPrimOp(Multiply(), Cast(two, TFloat64), pi), r)
        },
        ApplyBinaryPrimOp(
          Multiply(),
          ApplyBinaryPrimOp(Multiply(), Cast(I32(2), TFloat64), F64(pi)),
          r,
        ),
        """Forward constant primitive values and simple use ref.""",
      ),
      (
        IRBuilder.scoped { b =>
          val x0 = b.strictMemoize(I32(2))
          val x1 = b.strictMemoize(Cast(x0, TFloat64))
          val x2 = b.strictMemoize(ApplyBinaryPrimOp(FloatingPointDivide(), x1, F64(2)))
          val x3 = b.strictMemoize(F64(pi))
          val x4 = b.strictMemoize(ApplyBinaryPrimOp(Multiply(), x3, x1))
          val x5 = b.strictMemoize(ApplyBinaryPrimOp(Multiply(), x2, x2))
          val x6 = b.strictMemoize(ApplyBinaryPrimOp(Multiply(), x3, x5))
          MakeStruct(FastSeq("radius" -> x2, "circumference" -> x4, "area" -> x6))
        },
        IRBuilder.scoped { b =>
          val x1 = b.strictMemoize(Cast(I32(2), TFloat64))
          val x2 = b.strictMemoize(ApplyBinaryPrimOp(FloatingPointDivide(), x1, F64(2)))
          MakeStruct(FastSeq(
            "radius" -> x2,
            "circumference" -> ApplyBinaryPrimOp(Multiply(), F64(pi), x1),
            "area" -> ApplyBinaryPrimOp(
              Multiply(),
              F64(pi),
              ApplyBinaryPrimOp(Multiply(), x2, x2),
            ),
          ))
        },
        "Cascading Let-bindings are forwarded",
      ),
    )
  }

  @ParameterizedTest
  def testTrivialCases(input: IR, _expected: IR, reason: String)(implicit ctx: ExecuteContext)
    : Unit = {
    val normalize: (ExecuteContext, BaseIR) => BaseIR = NormalizeNames(allowFreeVariables = true)
    val result = normalize(ctx, ForwardLets(ctx, input))
    val expected = normalize(ctx, _expected)
    assertEq(
      result,
      normalize(ctx, expected),
      s"\ninput:\n${Pretty.sexprStyle(input)}\nexpected:\n${Pretty.sexprStyle(expected)}\ngot:\n${Pretty.sexprStyle(result)}\n$reason",
    )
  }

  @Test def testAggregators(implicit ctx: ExecuteContext): Unit = {
    val row = Ref(freshName(), TStruct("idx" -> TInt32))
    val aggEnv = Env[Type](row.name -> row.typ)

    val ir0 = ApplyAggOp(Sum())(bindIR(GetField(row, "idx") - 1)(x => Cast(x, TFloat64)))

    TypeCheck(ctx, ForwardLets(ctx, ir0), BindingEnv(Env.empty, agg = Some(aggEnv)))
  }

  @Test def testNestedBindingOverwrites(implicit ctx: ExecuteContext): Unit = {
    val x = Ref(freshName(), TInt32)
    val env = Env[Type](x.name -> TInt32)
    def xCast = Cast(x, TFloat64)
    val ir = bindIRs(xCast, xCast) { case Seq(x1, x2) => x2 + x2 + x1 }

    TypeCheck(ctx, ir, BindingEnv(env))
    TypeCheck(ctx, ForwardLets(ctx, ir), BindingEnv(env))
  }

  @Test def testLetsDoNotForwardInsideArrayAggWithNoOps(implicit ctx: ExecuteContext): Unit = {
    val y = Ref(freshName(), TInt32)
    val x = bindIR(
      streamAggIR(ToStream(In(0, TArray(TInt32))))(_ => y)
    )(x => streamAggIR(ToStream(In(1, TArray(TInt32))))(_ => y + x))

    TypeCheck(ctx, x, BindingEnv(Env(y.name -> TInt32)))
    TypeCheck(ctx, ForwardLets(ctx, x), BindingEnv(Env(y.name -> TInt32)))
  }
}
