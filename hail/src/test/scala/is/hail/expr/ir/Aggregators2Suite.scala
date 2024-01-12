package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.agg._
import is.hail.io.BufferSpec
import is.hail.types.{tcoerce, MatrixType, VirtualTypeWithReq}
import is.hail.types.physical._
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.{Call0, Call1, Call2}

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class Aggregators2Suite extends HailSuite {

  def assertAggEqualsProcessed(
    aggSig: PhysicalAggSig,
    initOp: IR,
    seqOps: IndexedSeq[IR],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastSeq(),
    nPartitions: Int = 2,
    expectedInit: Option[Any] = None,
    transformResult: Option[Any => Any] = None,
  ): Unit = {
    assert(seqOps.length >= 2 * nPartitions, s"Test aggregators with a larger stream!")

    val argT = PType.canonical(
      TStruct(args.map { case (n, (typ, _)) => n -> typ }: _*)
    ).setRequired(true).asInstanceOf[PStruct]
    val argVs = Row.fromSeq(args.map { case (_, (_, v)) => v })
    val argRef = Ref(genUID(), argT.virtualType)
    val spec = BufferSpec.wireSpec

    val (_, combAndDuplicate) = CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      Array.fill(nPartitions)(aggSig.state),
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      Begin(
        Array.tabulate(nPartitions)(i => DeserializeAggs(i, i, spec, Array(aggSig.state))) ++
          Array.range(1, nPartitions).map(i => CombOp(0, i, aggSig)) :+
          SerializeAggs(0, 0, spec, Array(aggSig.state)) :+
          DeserializeAggs(1, 0, spec, Array(aggSig.state))
      ),
    )

    val (Some(PTypeReferenceSingleCodeType(rt: PTuple)), resF) =
      CompileWithAggregators[AsmFunction1RegionLong](
        ctx,
        Array.fill(nPartitions)(aggSig.state),
        FastSeq(),
        FastSeq(classInfo[Region]),
        LongInfo,
        ResultOp.makeTuple(Array(aggSig, aggSig)),
      )
    assert(rt.types(0) == rt.types(1))

    val resultType = rt.types(0)
    if (transformResult.isEmpty)
      assert(
        resultType.virtualType.typeCheck(expected),
        s"expected type ${resultType.virtualType.parsableString()}, got $expected",
      )

    pool.scopedRegion { region =>
      val argOff = ScalaToRegionValue(ctx.stateManager, region, argT, argVs)

      def withArgs(foo: IR) = {
        CompileWithAggregators[AsmFunction2RegionLongUnit](
          ctx,
          Array(aggSig.state),
          FastSeq((argRef.name, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argT)))),
          FastSeq(classInfo[Region], LongInfo),
          UnitInfo,
          Let(args.map { case (n, _) => n -> GetField(argRef, n) }, foo),
        )._2
      }

      val serialize = SerializeAggs(0, 0, spec, Array(aggSig.state))
      val (_, writeF) = CompileWithAggregators[AsmFunction1RegionUnit](
        ctx,
        Array(aggSig.state),
        FastSeq(),
        FastSeq(classInfo[Region]),
        UnitInfo,
        serialize,
      )

      val initF = withArgs(initOp)

      expectedInit.foreach { v =>
        val (Some(PTypeReferenceSingleCodeType(rt: PBaseStruct)), resOneF) =
          CompileWithAggregators[AsmFunction1RegionLong](
            ctx,
            Array(aggSig.state),
            FastSeq(),
            FastSeq(classInfo[Region]),
            LongInfo,
            ResultOp.makeTuple(Array(aggSig)),
          )

        val init = initF(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        val res = resOneF(theHailClassLoader, ctx.fs, ctx.taskContext, region)

        pool.scopedSmallRegion { aggRegion =>
          init.newAggState(aggRegion)
          init(region, argOff)
          res.setAggState(aggRegion, init.getAggOffset())
          val result = SafeRow(rt, res(region)).get(0)
          assert(resultType.virtualType.valuesSimilar(result, v))
        }
      }

      val serializedParts =
        seqOps.grouped(math.ceil(seqOps.length / nPartitions.toDouble).toInt).map { seqs =>
          val init = initF(theHailClassLoader, ctx.fs, ctx.taskContext, region)
          val seq = withArgs(Begin(seqs))(theHailClassLoader, ctx.fs, ctx.taskContext, region)
          val write = writeF(theHailClassLoader, ctx.fs, ctx.taskContext, region)
          pool.scopedSmallRegion { aggRegion =>
            init.newAggState(aggRegion)
            init(region, argOff)
            val ioff = init.getAggOffset()
            seq.setAggState(aggRegion, ioff)
            seq(region, argOff)
            val soff = seq.getAggOffset()
            write.setAggState(aggRegion, soff)
            write(region)
            write.getSerializedAgg(0)
          }
        }.toArray

      pool.scopedSmallRegion { aggRegion =>
        val combOp = combAndDuplicate(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        combOp.newAggState(aggRegion)
        serializedParts.zipWithIndex.foreach { case (s, i) =>
          combOp.setSerializedAgg(i, s)
        }
        combOp(region)
        val res = resF(theHailClassLoader, ctx.fs, ctx.taskContext, region)
        res.setAggState(aggRegion, combOp.getAggOffset())
        val double = SafeRow(rt, res(region))
        transformResult match {
          case Some(f) =>
            assert(
              f(double.get(0)) == f(double.get(1)),
              s"\nbefore: ${f(double.get(0))}\nafter:  ${f(double.get(1))}",
            )
            assert(
              f(double.get(0)) == expected,
              s"\nresult: ${f(double.get(0))}\nexpect: $expected",
            )
          case None =>
            assert(
              resultType.virtualType.valuesSimilar(
                double.get(0),
                double.get(1),
              ), // state does not change through serialization
              s"\nbefore: ${double.get(0)}\nafter:  ${double.get(1)}",
            )
            assert(
              resultType.virtualType.valuesSimilar(double.get(0), expected),
              s"\nresult: ${double.get(0)}\nexpect: $expected",
            )
        }
      }
    }
  }

  def assertAggEquals(
    aggSig: PhysicalAggSig,
    initArgs: IndexedSeq[IR],
    seqArgs: IndexedSeq[IndexedSeq[IR]],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastSeq(),
    nPartitions: Int = 2,
    expectedInit: Option[Any] = None,
    transformResult: Option[Any => Any] = None,
  ): Unit =
    assertAggEqualsProcessed(
      aggSig,
      InitOp(0, initArgs, aggSig),
      seqArgs.map(s => SeqOp(0, s, aggSig)),
      expected,
      args,
      nPartitions,
      expectedInit,
      transformResult,
    )

  val t = TStruct("a" -> TString, "b" -> TInt64)
  val rows = FastSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
  val arrayType = TArray(t)

  val pnnAggSig = PhysicalAggSig(PrevNonnull(), TypedStateSig(VirtualTypeWithReq.fullyOptional(t)))

  val countAggSig = PhysicalAggSig(
    Count(),
    TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)),
  )

  val sumAggSig =
    PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)))

  def collectAggSig(t: Type): PhysicalAggSig =
    PhysicalAggSig(Collect(), CollectStateSig(VirtualTypeWithReq(PType.canonical(t))))

  @Test def TestCount() {
    val seqOpArgs = Array.fill(rows.length)(FastSeq[IR]())
    assertAggEquals(
      countAggSig,
      FastSeq(),
      seqOpArgs,
      expected = rows.length.toLong,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testSum() {
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b"))
    )
    assertAggEquals(
      sumAggSig,
      FastSeq(),
      seqOpArgs,
      expected = 10L,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testPrevNonnullStr() {
    val aggSig =
      PhysicalAggSig(PrevNonnull(), TypedStateSig(VirtualTypeWithReq(PCanonicalString())))
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a"))
    )

    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgs,
      expected = rows.last.get(0),
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testPrevNonnull() {
    val seqOpArgs =
      Array.tabulate(rows.length)(i => FastSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))
    assertAggEquals(
      pnnAggSig,
      FastSeq(),
      seqOpArgs,
      expected = rows.last,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testProduct() {
    val aggSig = PhysicalAggSig(
      Product(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)),
    )
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b"))
    )
    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgs,
      expected = -70L,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testCallStats() {
    val t = TStruct("x" -> TCall)

    val calls = FastSeq(
      Row(Call0()),
      Row(Call1(0)),
      Row(Call1(1)),
      Row(Call1(2)),
      Row(Call1(0)),
      null,
      null,
      Row(Call2(0, 0)),
      Row(Call2(0, 0, phased = true)),
      Row(Call2(0, 0)),
      Row(Call2(0, 1)),
      Row(Call2(1, 0, phased = true)),
      Row(Call2(1, 1)),
      Row(Call2(1, 3)),
      null,
      null,
      Row(null),
    )

    val aggSig = PhysicalAggSig(CallStats(), CallStatsStateSig())

    def seqOpArgs(calls: IndexedSeq[Any]) = Array.tabulate(calls.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("calls", TArray(t)), i), "x"))
    )

    val an = 18
    val ac = FastSeq(10, 6, 1, 1, 0)
    val af = ac.map(_.toDouble / an).toFastSeq
    val homCount = FastSeq(3, 1, 0, 0, 0)
    assertAggEquals(
      aggSig,
      FastSeq(I32(5)),
      seqOpArgs(calls),
      expected = Row(ac, af, an, homCount),
      args = FastSeq(("calls", (TArray(t), calls))),
    )

    val allMissing = calls.filter(_ == null)
    assertAggEquals(
      aggSig,
      FastSeq(I32(5)),
      seqOpArgs(allMissing),
      expected = Row(FastSeq(0, 0, 0, 0, 0), null, 0, FastSeq(0, 0, 0, 0, 0)),
      args = FastSeq(("calls", (TArray(t), allMissing))),
    )
  }

  @Test def testTakeBy() {
    val t = TStruct(
      "a" -> TStruct("x" -> TInt32, "y" -> TInt64),
      "b" -> TInt32,
      "c" -> TInt64,
      "d" -> TFloat32,
      "e" -> TFloat64,
      "f" -> TBoolean,
      "g" -> TString,
      "h" -> TArray(TInt32),
    )

    val rows = FastSeq(
      Row(Row(11, 11L), 1, 1L, 1f, 1d, true, "1", FastSeq(1, 1)),
      Row(Row(22, 22L), 2, 2L, 2f, 2d, false, "11", null),
      Row(Row(33, 33L), 3, 3L, 3f, 3d, null, "111", FastSeq(3, null)),
      Row(Row(44, null), 4, 4L, 4f, 4d, true, "1111", null),
      Row(Row(55, null), 5, 5L, 5f, 5d, true, "11111", null),
      Row(Row(66, 66L), 6, 6L, 6f, 6d, false, "111111", FastSeq(6, 6, 6, 6)),
      Row(Row(77, 77L), 7, 7L, 7f, 7d, false, "1111111", FastSeq()),
      Row(Row(88, 88L), 8, 8L, 8f, 8d, null, "11111111", null),
      Row(Row(99, 99L), 9, 9L, 9f, 9d, null, "111111111", FastSeq(null)),
      Row(Row(1010, 1010L), 10, 10L, 10f, 10d, false, "1111111111", FastSeq()),
      Row(Row(1010, 1011L), 11, 11L, 11f, 11d, true, "11111111111", FastSeq()),
      Row(null, null, null, null, null, null, null, null),
      Row(null, null, null, null, null, null, null, null),
      Row(null, null, null, null, null, null, null, null),
    )

    val rowsReversed = rows.take(rows.length - 3).reverse ++ rows.takeRight(3)

    val permutations = Array(
      rows, // sorted
      rows.reverse, // reversed
      rows.take(6).reverse ++ rows.drop(6), // down and up
      rows.drop(6) ++ rows.take(6).reverse, // up and down
      {
        val (a, b) = rows.zipWithIndex.partition(_._2 % 2 == 0)
        a.map(_._1) ++ b.map(_._1)
      }, // random-ish
    )

    val valueTransformations: Array[(Type, IR => IR, Row => Any)] = Array(
      (t, identity[IR], identity[Row]),
      (TInt32, GetField(_, "b"), Option(_).map(_.get(1)).orNull),
      (TFloat64, GetField(_, "e"), Option(_).map(_.get(4)).orNull),
      (TBoolean, GetField(_, "f"), Option(_).map(_.get(5)).orNull),
      (TString, GetField(_, "g"), Option(_).map(_.get(6)).orNull),
      (TArray(TInt32), GetField(_, "h"), Option(_).map(_.get(7)).orNull),
    )

    val keyTransformations: Array[(Type, IR => IR)] = Array(
      (TInt32, GetField(_, "b")),
      (TFloat64, GetField(_, "e")),
      (TString, GetField(_, "g")),
      (TStruct("x" -> TInt32, "y" -> TInt64), GetField(_, "a")),
    )

    def test(
      n: Int,
      data: IndexedSeq[Row],
      valueType: Type,
      valueF: IR => IR,
      resultF: Row => Any,
      keyType: Type,
      keyF: IR => IR,
      so: SortOrder = Ascending,
    ): Unit = {

      val aggSig = PhysicalAggSig(
        TakeBy(),
        TakeByStateSig(
          VirtualTypeWithReq(PType.canonical(valueType)),
          VirtualTypeWithReq(PType.canonical(keyType)),
          so,
        ),
      )
      val seqOpArgs = Array.tabulate(rows.length) { i =>
        val ref = ArrayRef(Ref("rows", TArray(t)), i)
        FastSeq[IR](valueF(ref), keyF(ref))
      }

      assertAggEquals(
        aggSig,
        FastSeq(I32(n)),
        seqOpArgs,
        expected = (if (so == Descending) rowsReversed else rows).take(n).map(resultF),
        args = FastSeq(("rows", (TArray(t), data))),
      )
    }

    // test counts and data input orderings
    for {
      n <- FastSeq(0, 1, 4, 100)
      perm <- permutations
      so <- FastSeq(Ascending, Descending)
    }
      test(n, perm, t, identity[IR], identity[Row], TInt32, GetField(_, "b"), so)

    // test key and value types
    for {
      (vt, valueF, resultF) <- valueTransformations
      (kt, keyF) <- keyTransformations
    }
      test(4, permutations.last, vt, valueF, resultF, kt, keyF)

    // test stable sort
    test(7, rows, t, identity[IR], identity[Row], TInt64, _ => I64(5L))

    // test GC behavior by passing a large collection
    val rows2 = Array.tabulate(1200)(i => Row(i, i.toString)).toFastSeq
    val t2 = TStruct("a" -> TInt32, "b" -> TString)
    val aggSig2 = PhysicalAggSig(
      TakeBy(),
      TakeByStateSig(
        VirtualTypeWithReq(PType.canonical(t2)),
        VirtualTypeWithReq(PType.canonical(TInt32)),
        Ascending,
      ),
    )
    val seqOpArgs2 = Array.tabulate(rows2.length)(i =>
      FastSeq[IR](
        ArrayRef(Ref("rows", TArray(t2)), i),
        GetField(ArrayRef(Ref("rows", TArray(t2)), i), "a"),
      )
    )

    assertAggEquals(
      aggSig2,
      FastSeq(I32(17)),
      seqOpArgs2,
      expected = rows2.take(17),
      args = FastSeq(("rows", (TArray(t2), rows2.reverse))),
    )

    // test inside of aggregation
    val tr = TableRange(10000, 5)
    val ta = TableAggregate(
      tr,
      ApplyAggOp(
        FastSeq(19),
        FastSeq(
          invoke("str", TString, GetField(Ref("row", tr.typ.rowType), "idx")),
          I32(9999) - GetField(Ref("row", tr.typ.rowType), "idx"),
        ),
        AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(TString, TInt32)),
      ),
    )

    assertEvalsTo(ta, (0 until 19).map(i => (9999 - i).toString).toFastSeq)(
      ExecStrategy.interpretOnly
    )
  }

  @Test def testTake() {
    val t = TStruct(
      "a" -> TStruct("x" -> TInt32, "y" -> TInt64),
      "b" -> TInt32,
      "c" -> TInt64,
      "d" -> TFloat32,
      "e" -> TFloat64,
      "f" -> TBoolean,
      "g" -> TString,
      "h" -> TArray(TInt32),
    )

    val rows = FastSeq(
      Row(Row(11, 11L), 1, 1L, 1f, 1d, true, "one", FastSeq(1, 1)),
      Row(Row(22, 22L), 2, 2L, 2f, 2d, false, "two", null),
      null,
      Row(Row(33, 33L), 3, 3L, 3f, 3d, null, "three", FastSeq(3, null)),
      Row(null, null, null, null, null, null, null, FastSeq()),
      Row(Row(null, 44L), 4, 4L, 4f, 4d, true, "four", null),
      Row(Row(55, null), 5, 5L, 5f, 5d, true, null, null),
      null,
      Row(Row(66, 66L), 6, 6L, 6f, 6d, false, "six", FastSeq(6, 6, 6, 6)),
      Row(null, null, null, null, null, null, null, null),
      Row(Row(77, 77L), 7, 7L, 7f, 7d, false, "seven", FastSeq()),
      null,
      null,
      Row(null, null, null, null, null, null, null, null),
      Row(Row(88, 88L), 8, 8L, 8f, 8d, null, "eight", null),
      Row(Row(99, 99L), 9, 9L, 9f, 9d, null, "nine", FastSeq(null)),
      Row(Row(1010, 1010L), 10, 10L, 10f, 10d, false, "ten", FastSeq()),
      Row(Row(1111, 1111L), 11, 11L, 11f, 11d, true, "eleven", FastSeq()),
    )

    val aggSig = PhysicalAggSig(Take(), TakeStateSig(VirtualTypeWithReq(PType.canonical(t))))
    val seqOpArgs =
      Array.tabulate(rows.length)(i => FastSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    FastSeq(0, 1, 3, 8, 10, 15, 30).foreach { i =>
      assertAggEquals(
        aggSig,
        FastSeq(I32(i)),
        seqOpArgs,
        expected = rows.take(i),
        args = FastSeq(("rows", (TArray(t), rows))),
      )
    }

    val transformations: IndexedSeq[(IR => IR, Row => Any, Type)] = t.fields.map { f =>
      ((x: IR) => GetField(x, f.name), (r: Row) => if (r == null) null else r.get(f.index), f.typ)
    }.filter(_._3 == TString)

    transformations.foreach { case (irF, rowF, subT) =>
      val aggSig = PhysicalAggSig(Take(), TakeStateSig(VirtualTypeWithReq(PType.canonical(subT))))
      val seqOpArgs =
        Array.tabulate(rows.length)(i => FastSeq[IR](irF(ArrayRef(Ref("rows", TArray(t)), i))))

      val expected = rows.take(10).map(rowF)
      assertAggEquals(
        aggSig,
        FastSeq(I32(10)),
        seqOpArgs,
        expected = expected,
        args = FastSeq(("rows", (TArray(t), rows))),
      )
    }
  }

  def seqOpOverArray(aggIdx: Int, a: IR, seqOps: IR => IR, alstate: ArrayLenAggSig): IR = {
    val idx = Ref(genUID(), TInt32)

    Begin(FastSeq(
      SeqOp(aggIdx, FastSeq(ArrayLen(a)), alstate),
      StreamFor(
        StreamRange(0, ArrayLen(a), 1),
        idx.name,
        bindIR(ArrayRef(a, idx)) { elt =>
          SeqOp(aggIdx, FastSeq(idx, seqOps(elt)), AggElementsAggSig(alstate.nested))
        },
      ),
    ))
  }

  @Test def testMin() {
    val aggSig = PhysicalAggSig(Min(), TypedStateSig(VirtualTypeWithReq(PInt64(false))))
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b"))
    )
    val seqOpArgsNA = Array.tabulate(8)(i => FastSeq[IR](NA(TInt64)))

    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgs,
      expected = -2L,
      args = FastSeq(("rows", (arrayType, rows))),
    )
    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgsNA,
      expected = null,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testMax() {
    val aggSig = PhysicalAggSig(Max(), TypedStateSig(VirtualTypeWithReq(PInt64(false))))
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b"))
    )
    val seqOpArgsNA = Array.tabulate(8)(i => FastSeq[IR](NA(TInt64)))

    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgs,
      expected = 7L,
      args = FastSeq(("rows", (arrayType, rows))),
    )
    assertAggEquals(
      aggSig,
      FastSeq(),
      seqOpArgsNA,
      expected = null,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testCollectLongs() {
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b"))
    )
    assertAggEquals(
      collectAggSig(TInt64),
      FastSeq(),
      seqOpArgs,
      expected = FastSeq(5L, null, -2L, 7L, null, null),
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testCollectStrs() {
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a"))
    )

    assertAggEquals(
      collectAggSig(TString),
      FastSeq(),
      seqOpArgs,
      expected = FastSeq("abcd", null, null, "abcd", null, "foo"),
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testCollectBig() {
    val seqOpArgs = Array.tabulate(100)(i => FastSeq(I64(i)))
    assertAggEquals(
      collectAggSig(TInt64),
      FastSeq(),
      seqOpArgs,
      expected = Array.tabulate(100)(i => i.toLong).toIndexedSeq,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testArrayElementsAgg() {
    val alState = ArrayLenAggSig(knownLength = false, FastSeq(pnnAggSig, countAggSig, sumAggSig))

    val value = FastSeq(
      FastSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastSeq(null, null, null, Row("f", 5L)),
    )

    val expected =
      FastSeq(
        Row(Row("a", 4L), 6L, 10L),
        Row(Row("b", 4L), 6L, 9L),
        Row(Row("c", 4L), 6L, 8L),
        Row(Row("f", 5L), 6L, 10L),
      )

    val init = InitOp(
      0,
      FastSeq(Begin(FastSeq[IR](
        InitOp(0, FastSeq(), pnnAggSig),
        InitOp(1, FastSeq(), countAggSig),
        InitOp(2, FastSeq(), sumAggSig),
      ))),
      alState,
    )

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(
        0,
        ArrayRef(stream, i),
        elt =>
          Begin(FastSeq(
            SeqOp(0, FastSeq(elt), pnnAggSig),
            SeqOp(1, FastSeq(), countAggSig),
            SeqOp(2, FastSeq(GetField(elt, "b")), sumAggSig),
          )),
        alState,
      )
    }

    assertAggEqualsProcessed(
      alState,
      init,
      seq,
      expected,
      FastSeq(("stream", (stream.typ, value))),
      2,
      None,
    )
  }

  @Test def testNestedArrayElementsAgg() {
    val alstate1 = ArrayLenAggSig(knownLength = false, FastSeq(sumAggSig))
    val aestate1 = AggElementsAggSig(FastSeq(sumAggSig))
    val alstate2 = ArrayLenAggSig(knownLength = false, FastSeq[PhysicalAggSig](alstate1))

    val init = InitOp(
      0,
      FastSeq(Begin(FastSeq[IR](
        InitOp(
          0,
          FastSeq(Begin(FastSeq[IR](
            InitOp(0, FastSeq(), sumAggSig)
          ))),
          alstate1,
        )
      ))),
      alstate2,
    )

    val stream = Ref("stream", TArray(TArray(TArray(TInt64))))
    val seq = Array.tabulate(10) { i =>
      seqOpOverArray(
        0,
        ArrayRef(stream, i),
        array1 => seqOpOverArray(0, array1, elt => SeqOp(0, FastSeq(elt), sumAggSig), alstate1),
        alstate2,
      )
    }

    val expected = FastSeq(Row(FastSeq(Row(45L))))

    val args = Array.tabulate(10)(i => FastSeq(FastSeq(i.toLong))).toFastSeq
    assertAggEqualsProcessed(
      alstate2,
      init,
      seq,
      expected,
      FastSeq(("stream", (stream.typ, args))),
      2,
      None,
    )
  }

  @Test def testArrayElementsAggTake() {
    val value = FastSeq(
      FastSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastSeq(null, null, null, Row("f", 5L)),
    )

    val take = PhysicalAggSig(Take(), TakeStateSig(VirtualTypeWithReq(PType.canonical(t))))
    val alstate = ArrayLenAggSig(knownLength = false, FastSeq(take))

    val init = InitOp(
      0,
      FastSeq(Begin(FastSeq[IR](
        InitOp(0, FastSeq(I32(3)), take)
      ))),
      alstate,
    )

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), elt => SeqOp(0, FastSeq(elt), take), alstate)
    }

    val expected = Array.tabulate(value(0).length)(i =>
      Row(Array.tabulate(3)(j => value(j)(i)).toFastSeq)
    ).toFastSeq
    assertAggEqualsProcessed(
      alstate,
      init,
      seq,
      expected,
      FastSeq(("stream", (stream.typ, value))),
      2,
      None,
    )
  }

  @Test def testGroup() {
    val group = GroupedAggSig(
      VirtualTypeWithReq(PCanonicalString()),
      FastSeq(pnnAggSig, countAggSig, sumAggSig),
    )

    val initOpArgs = FastSeq(Begin(FastSeq(
      InitOp(0, FastSeq(), pnnAggSig),
      InitOp(1, FastSeq(), countAggSig),
      InitOp(2, FastSeq(), sumAggSig),
    )))

    val rows =
      FastSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](
        GetField(ArrayRef(rref, i), "a"),
        Begin(FastSeq(
          SeqOp(0, FastSeq(ArrayRef(rref, i)), pnnAggSig),
          SeqOp(1, FastSeq(), countAggSig),
          SeqOp(2, FastSeq(GetField(ArrayRef(rref, i), "b")), sumAggSig),
        )),
      )
    )

    val expected = Map(
      "abcd" -> Row(Row("abcd", 7L), 2L, 12L),
      "foo" -> Row(Row("foo", null), 1L, 0L),
      (null, Row(Row(null, -2L), 3L, -2L)),
    )

    assertAggEquals(
      group,
      initOpArgs,
      seqOpArgs,
      expected = expected,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testNestedGroup() {

    val group1 = GroupedAggSig(
      VirtualTypeWithReq(PCanonicalString()),
      FastSeq(pnnAggSig, countAggSig, sumAggSig),
    )
    val group2 =
      GroupedAggSig(VirtualTypeWithReq(PCanonicalString()), FastSeq[PhysicalAggSig](group1))

    val initOpArgs = FastSeq(
      InitOp(
        0,
        FastSeq(
          Begin(FastSeq(
            InitOp(0, FastSeq(), pnnAggSig),
            InitOp(1, FastSeq(), countAggSig),
            InitOp(2, FastSeq(), sumAggSig),
          ))
        ),
        group1,
      )
    )

    val rows =
      FastSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](
        GetField(ArrayRef(rref, i), "a"),
        SeqOp(
          0,
          FastSeq[IR](
            GetField(ArrayRef(rref, i), "a"),
            Begin(FastSeq(
              SeqOp(0, FastSeq(ArrayRef(rref, i)), pnnAggSig),
              SeqOp(1, FastSeq(), countAggSig),
              SeqOp(2, FastSeq(GetField(ArrayRef(rref, i), "b")), sumAggSig),
            )),
          ),
          group1,
        ),
      )
    )

    val expected = Map(
      "abcd" -> Row(Map("abcd" -> Row(Row("abcd", 7L), 2L, 12L))),
      "foo" -> Row(Map("foo" -> Row(Row("foo", null), 1L, 0L))),
      (null, Row(Map((null, Row(Row(null, -2L), 3L, -2L))))),
    )

    assertAggEquals(
      group2,
      initOpArgs,
      seqOpArgs,
      expected = expected,
      args = FastSeq(("rows", (arrayType, rows))),
    )
  }

  @Test def testCollectAsSet() {
    val rows =
      FastSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))
    val elts = Array.tabulate(rows.length)(i => FastSeq(GetField(ArrayRef(rref, i), "a")))
    val eltsPrimitive = Array.tabulate(rows.length)(i => FastSeq(GetField(ArrayRef(rref, i), "b")))

    val expected = Set("abcd", "foo", null)
    val expectedPrimitive = Set(5L, -2L, 7L, null)

    val aggsig =
      PhysicalAggSig(CollectAsSet(), CollectAsSetStateSig(VirtualTypeWithReq(PCanonicalString())))
    val aggsigPrimitive =
      PhysicalAggSig(CollectAsSet(), CollectAsSetStateSig(VirtualTypeWithReq(PInt64())))
    assertAggEquals(
      aggsig,
      FastSeq(),
      elts,
      expected = expected,
      args = FastSeq(("rows", (arrayType, rows))),
      expectedInit = Some(Set()),
    )
    assertAggEquals(
      aggsigPrimitive,
      FastSeq(),
      eltsPrimitive,
      expected = expectedPrimitive,
      args = FastSeq(("rows", (arrayType, rows))),
      expectedInit = Some(Set()),
    )
  }

  @Test def testDownsample() {
    val aggSig = PhysicalAggSig(
      Downsample(),
      DownsampleStateSig(VirtualTypeWithReq(PCanonicalArray(PCanonicalString()))),
    )
    val rows = FastSeq(
      Row(-1.23, 1.23, null),
      Row(-10d, 10d, FastSeq("foo")),
      Row(0d, 100d, FastSeq()),
      Row(0d, 100d, FastSeq()),
      Row(-10.1d, -100d, null),
      Row(0d, 0d, null),
      Row(0d, 0d, null),
      Row(1d, 1.1d, null),
      Row(1d, 1.1d, null),
      Row(2d, 2.2d, null),
      Row(3d, 3.3d, null),
      Row(3d, 3.3d, null),
      Row(3d, 3.3d, null),
      Row(4d, 4.4d, null),
      Row(3d, 3.3d, null),
      Row(3d, 3.3d, null),
      Row(3d, 3.3d, null),
    )

    val arrayType = TArray(TStruct("x" -> TFloat64, "y" -> TFloat64, "label" -> TArray(TString)))
    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastSeq[IR](
        GetField(ArrayRef(Ref("rows", arrayType), i), "x"),
        GetField(ArrayRef(Ref("rows", arrayType), i), "y"),
        GetField(ArrayRef(Ref("rows", arrayType), i), "label"),
      )
    )

    assertAggEquals(
      aggSig,
      FastSeq(I32(500)),
      Array.fill[IndexedSeq[IR]](20)(FastSeq(NA(TFloat64), NA(TFloat64), NA(TArray(TString)))),
      expected = FastSeq(),
      args = FastSeq(("rows", (arrayType, rows))),
    )

    val expected = rows.toSet
    assertAggEquals(
      aggSig,
      FastSeq(I32(100)),
      seqOpArgs,
      expected = expected,
      args = FastSeq(("rows", (arrayType, rows))),
      transformResult = Some(_.asInstanceOf[IndexedSeq[_]].toSet),
    )
  }

  @Test def testLoweringMatrixMapColsWithAggFilterAndLets(): Unit = {
    val t = MatrixType(
      TStruct.empty,
      FastSeq("col_idx"),
      TStruct("col_idx" -> TInt32),
      FastSeq("row_idx"),
      TStruct("row_idx" -> TInt32),
      TStruct.empty,
    )
    val ir = TableCollect(MatrixColsTable(MatrixMapCols(
      MatrixRead(t, false, false, MatrixRangeReader(10, 10, None)),
      InsertFields(
        Ref("sa", t.colType),
        FastSeq((
          "foo",
          Let(
            FastSeq("bar" -> (GetField(Ref("sa", t.colType), "col_idx") + I32(1))),
            AggFilter(
              GetField(Ref("va", t.rowType), "row_idx") < I32(5),
              Ref("bar", TInt32).toL + Ref("bar", TInt32).toL + ApplyAggOp(
                FastSeq(),
                FastSeq(GetField(Ref("va", t.rowType), "row_idx").toL),
                AggSignature(Sum(), FastSeq(), FastSeq(TInt64)),
              ),
              false,
            ),
          ),
        )),
      ),
      Some(FastSeq()),
    )))
    assertEvalsTo(ir, Row((0 until 10).map(i => Row(i, 2L * i + 12L)), Row()))(
      ExecStrategy.interpretOnly
    )
  }

  @Test def testRunAggScan(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = PhysicalAggSig(
      Sum(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TFloat64).setRequired(true)),
    )
    val x = ToArray(RunAggScan(
      StreamRange(I32(0), I32(5), I32(1)),
      "foo",
      InitOp(0, FastSeq(), sig),
      SeqOp(0, FastSeq(Ref("foo", TInt32).toD), sig),
      ResultOp(0, sig),
      Array(sig.state),
    ))
    assertEvalsTo(x, FastSeq(0.0, 0.0, 1.0, 3.0, 6.0))
  }

  @Test def testNestedRunAggScan(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = PhysicalAggSig(
      Sum(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TFloat64).setRequired(true)),
    )
    val x =
      ToArray(
        StreamFlatMap(
          StreamRange(I32(3), I32(6), I32(1)),
          "i",
          RunAggScan(
            StreamRange(I32(0), Ref("i", TInt32), I32(1)),
            "foo",
            InitOp(0, FastSeq(), sig),
            SeqOp(0, FastSeq(Ref("foo", TInt32).toD), sig),
            ResultOp(0, sig),
            Array(sig.state),
          ),
        )
      )
    assertEvalsTo(
      x,
      FastSeq(
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 3.0,
        0.0, 0.0, 1.0, 3.0, 6.0),
    )
  }

  @Test def testRunAggBasic(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq(PFloat64(true))))
    val x = RunAgg(
      Begin(FastSeq(
        InitOp(0, FastSeq(), sig),
        SeqOp(0, FastSeq(F64(1.0)), sig),
        SeqOp(0, FastSeq(F64(-5.0)), sig),
      )),
      ResultOp.makeTuple(FastSeq(sig)),
      FastSeq(sig.state),
    )
    assertEvalsTo(x, Row(-4.0))
  }

  @Test def testRunAggNested(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sumSig = PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq(PFloat64(true))))
    val takeSig = PhysicalAggSig(Take(), TakeStateSig(VirtualTypeWithReq(PFloat64(true))))
    val x = RunAgg(
      Begin(FastSeq(
        InitOp(0, FastSeq(I32(5)), takeSig),
        StreamFor(
          StreamRange(I32(0), I32(10), I32(1)),
          "foo",
          SeqOp(
            0,
            FastSeq(
              RunAgg(
                Begin(FastSeq(
                  InitOp(0, FastSeq(), sumSig),
                  SeqOp(0, FastSeq(F64(-1.0)), sumSig),
                  SeqOp(0, FastSeq(Ref("foo", TInt32).toD), sumSig),
                )),
                ResultOp(0, sumSig),
                FastSeq(sumSig.state),
              )
            ),
            takeSig,
          ),
        ),
      )),
      ResultOp(0, takeSig),
      FastSeq(takeSig.state),
    )
    assertEvalsTo(x, FastSeq(-1d, 0d, 1d, 2d, 3d))
  }

  @Test(enabled = false) def testAggStateAndCombOp(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val takeSig = PhysicalAggSig(Take(), TakeStateSig(VirtualTypeWithReq(PInt64(true))))
    val x = Let(
      FastSeq("x" -> RunAgg(
        Begin(FastSeq(
          InitOp(0, FastSeq(I32(10)), takeSig),
          SeqOp(0, FastSeq(NA(TInt64)), takeSig),
          SeqOp(0, FastSeq(I64(-1L)), takeSig),
          SeqOp(0, FastSeq(I64(2L)), takeSig),
        )),
        AggStateValue(0, takeSig.state),
        FastSeq(takeSig.state),
      )),
      RunAgg(
        Begin(FastSeq(
          InitOp(0, FastSeq(I32(10)), takeSig),
          CombOpValue(0, Ref("x", TBinary), takeSig),
          SeqOp(0, FastSeq(I64(3L)), takeSig),
          CombOpValue(0, Ref("x", TBinary), takeSig),
          SeqOp(0, FastSeq(I64(0L)), takeSig),
        )),
        ResultOp(0, takeSig),
        FastSeq(takeSig.state),
      ),
    )

    assertEvalsTo(x, FastSeq(null, -1L, 2L, 3L, null, null, -1L, 2L, 0L))
  }
}
