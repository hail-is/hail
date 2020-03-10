package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types.MatrixType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.utils._
import is.hail.variant.{Call0, Call1, Call2}
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class Aggregators2Suite extends HailSuite {

  def assertAggEqualsProcessed(
    aggSig: AggStateSignature,
    initOp: IR,
    seqOps: IndexedSeq[IR],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastIndexedSeq(),
    nPartitions: Int = 2,
    expectedInit: Option[Any] = None,
    transformResult: Option[Any => Any] = None
  ): Unit = {
    assert(seqOps.length >= 2 * nPartitions, s"Test aggregators with a larger stream!")

    val argT = PType.canonical(TStruct(args.map { case (n, (typ, _)) => n -> typ }: _*)).asInstanceOf[PStruct]
    val argVs = Row.fromSeq(args.map { case (_, (_, v)) => v })
    val argRef = Ref(genUID(), argT.virtualType)
    val spec = BufferSpec.defaultUncompressed

    def toCanonicalPhysical(aggSig: AggStateSignature): AggStatePhysicalSignature = {
        AggStatePhysicalSignature(
          aggSig.m.map { case (op, sig) =>
            (op, sig.toPhysical(sig.initOpArgs.map(PType.canonical), sig.seqOpArgs.map(PType.canonical)))
          },
          aggSig.default,
          aggSig.nested.map(_.map(toCanonicalPhysical))
        )
    }

    val psig = toCanonicalPhysical(aggSig)
    val (_, combAndDuplicate) = CompileWithAggregators2[Unit](ctx,
      Array.fill(nPartitions)(psig),
      Begin(
        Array.tabulate(nPartitions)(i => DeserializeAggs(i, i, spec, Array(aggSig))) ++
          Array.range(1, nPartitions).map(i => CombOp(0, i, aggSig)) :+
          SerializeAggs(0, 0, spec, Array(aggSig)) :+
          DeserializeAggs(1, 0, spec, Array(aggSig))))

    val (rt: PTuple, resF) = CompileWithAggregators2[Long](ctx,
      Array.fill(nPartitions)(psig),
      ResultOp(0, Array(aggSig, aggSig)))
    assert(rt.types(0) == rt.types(1))

    val resultType = rt.types(0)
    if (transformResult.isEmpty)
      assert(resultType.virtualType.typeCheck(expected), s"expected type ${ resultType.virtualType.parsableString() }, got ${expected}")

    Region.scoped { region =>
      val argOff = ScalaToRegionValue(region, argT, argVs)

      def withArgs(foo: IR) = {
        CompileWithAggregators2[Long, Unit](ctx,
          Array(psig),
          argRef.name, argT,
          args.map(_._1).foldLeft[IR](foo) { case (op, name) =>
            Let(name, GetField(argRef, name), op)
          })._2
      }

      val serialize = SerializeAggs(0, 0, spec, Array(aggSig))
      val (_, writeF) = CompileWithAggregators2[Unit](ctx,
        Array(psig),
        serialize)

      val initF = withArgs(initOp)

      expectedInit.foreach { v =>
        val (rt: PBaseStruct, resOneF) = CompileWithAggregators2[Long](ctx,
          Array(psig), ResultOp(0, Array(aggSig)))

        val init = initF(0, region)
        val res = resOneF(0, region)

        Region.smallScoped { aggRegion =>
          init.newAggState(aggRegion)
          init(region, argOff, false)
          res.setAggState(aggRegion, init.getAggOffset())
          val result = SafeRow(rt, res(region)).get(0)
          assert(resultType.virtualType.valuesSimilar(result, v))
        }
      }

      val serializedParts = seqOps.grouped(math.ceil(seqOps.length / nPartitions.toDouble).toInt).map { seqs =>
        val init = initF(0, region)
        val seq = withArgs(Begin(seqs))(0, region)
        val write = writeF(0, region)
        Region.smallScoped { aggRegion =>
          init.newAggState(aggRegion)
          init(region, argOff, false)
          val ioff = init.getAggOffset()
          seq.setAggState(aggRegion, ioff)
          seq(region, argOff, false)
          val soff = seq.getAggOffset()
          write.setAggState(aggRegion, soff)
          write(region)
          write.getSerializedAgg(0)
        }
      }.toArray

      Region.smallScoped { aggRegion =>
        val combOp = combAndDuplicate(0, region)
        combOp.newAggState(aggRegion)
        serializedParts.zipWithIndex.foreach { case (s, i) =>
          combOp.setSerializedAgg(i, s)
        }
        combOp(region)
        val res = resF(0, region)
        res.setAggState(aggRegion, combOp.getAggOffset())
        val double = SafeRow(rt, res(region))
        transformResult match {
          case Some(f) =>
            assert(f(double.get(0)) == f(double.get(1)),
              s"\nbefore: ${ f(double.get(0)) }\nafter:  ${ f(double.get(1)) }")
            assert(f(double.get(0)) == expected,
              s"\nresult: ${ f(double.get(0)) }\nexpect: ${ expected }")
          case None =>
            assert(resultType.virtualType.valuesSimilar(double.get(0), double.get(1)), // state does not change through serialization
              s"\nbefore: ${ double.get(0) }\nafter:  ${ double.get(1) }")
            assert(resultType.virtualType.valuesSimilar(double.get(0), expected),
              s"\nresult: ${ double.get(0) }\nexpect: $expected")
        }
      }
    }
  }

  def assertAggEquals(
    aggSig: AggStateSignature,
    initArgs: IndexedSeq[IR],
    seqArgs: IndexedSeq[IndexedSeq[IR]],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastIndexedSeq(),
    nPartitions: Int = 2,
    expectedInit: Option[Any] = None,
    transformResult: Option[Any => Any] = None): Unit =
    assertAggEqualsProcessed(aggSig,
      InitOp(0, initArgs, aggSig, aggSig.default),
      seqArgs.map(s => SeqOp(0, s, aggSig, aggSig.default)),
      expected, args, nPartitions, expectedInit,
      transformResult)

  val t = TStruct("a" -> TString, "b" -> TInt64)
  val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
  val arrayType = TArray(t)

  val pnnAggSig = AggSignature(PrevNonnull(), FastSeq(), FastSeq(t))
  val countAggSig = AggSignature(Count(), FastSeq(), FastSeq())
  val sumAggSig = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))

  def collectAggSig(t: Type): AggSignature = AggSignature(Collect(), FastSeq(), FastSeq(t))

  @Test def TestCount() {
    val aggSig = AggSignature(Count(), FastSeq(), FastSeq())
    val seqOpArgs = Array.fill(rows.length)(FastIndexedSeq[IR]())

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = rows.length.toLong, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testSum() {
    val aggSig = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = 10L, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testPrevNonnullStr() {
    val aggSig = AggSignature(PrevNonnull(), FastSeq(), FastSeq(TString))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a")))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = rows.last.get(0), args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testPrevNonnull() {
    val aggSig = AggSignature(PrevNonnull(), FastSeq(), FastSeq(t))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = rows.last, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testProduct() {
    val aggSig = AggSignature(Product(), FastSeq(), FastSeq(TInt64))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = -70L, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testCallStats() {
    val t = TStruct("x" -> TCall)

    val calls = FastIndexedSeq(
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
      Row(null))

    val aggSig = AggSignature(CallStats(), FastSeq(TInt32), FastSeq(TCall))

    def seqOpArgs(calls: IndexedSeq[Any]) = Array.tabulate(calls.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(Ref("calls", TArray(t)), i), "x")))

    val an = 18
    val ac = FastIndexedSeq(10, 6, 1, 1, 0)
    val af = ac.map(_.toDouble / an).toFastIndexedSeq
    val homCount = FastIndexedSeq(3, 1, 0, 0, 0)
    assertAggEquals(aggSig.singletonContainer,
      FastIndexedSeq(I32(5)),
      seqOpArgs(calls),
      expected = Row(ac, af, an, homCount),
      args = FastIndexedSeq(("calls", (TArray(t), calls))))

    val allMissing = calls.filter(_ == null)
    assertAggEquals(aggSig.singletonContainer,
      FastIndexedSeq(I32(5)),
      seqOpArgs(allMissing),
      expected = Row(FastIndexedSeq(0, 0, 0, 0, 0), null, 0, FastIndexedSeq(0, 0, 0, 0, 0)),
      args = FastIndexedSeq(("calls", (TArray(t), allMissing))))
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
      "h" -> TArray(TInt32))

    val rows = FastIndexedSeq(
      Row(Row(11, 11L), 1, 1L, 1f, 1d, true, "1", FastIndexedSeq(1, 1)),
      Row(Row(22, 22L), 2, 2L, 2f, 2d, false, "11", null),
      Row(Row(33, 33L), 3, 3L, 3f, 3d, null, "111", FastIndexedSeq(3, null)),
      Row(Row(44, null), 4, 4L, 4f, 4d, true, "1111", null),
      Row(Row(55, null), 5, 5L, 5f, 5d, true, "11111", null),
      Row(Row(66, 66L), 6, 6L, 6f, 6d, false, "111111", FastIndexedSeq(6, 6, 6, 6)),
      Row(Row(77, 77L), 7, 7L, 7f, 7d, false, "1111111", FastIndexedSeq()),
      Row(Row(88, 88L), 8, 8L, 8f, 8d, null, "11111111", null),
      Row(Row(99, 99L), 9, 9L, 9f, 9d, null, "111111111", FastIndexedSeq(null)),
      Row(Row(1010, 1010L), 10, 10L, 10f, 10d, false, "1111111111", FastIndexedSeq()),
      Row(Row(1010, 1011L), 11, 11L, 11f, 11d, true, "11111111111", FastIndexedSeq()),
      Row(null, null, null, null, null, null, null, null),
      Row(null, null, null, null, null, null, null, null),
      Row(null, null, null, null, null, null, null, null)
    )

    val permutations = Array(
      rows, // sorted
      rows.reverse, // reversed
      rows.take(6).reverse ++ rows.drop(6), // down and up
      rows.drop(6) ++ rows.take(6).reverse, // up and down
      {
        val (a, b) = rows.zipWithIndex.partition(_._2 % 2 == 0)
        a.map(_._1) ++ b.map(_._1)
      } // random-ish
    )

    val valueTransformations: Array[(Type, IR => IR, Row => Any)] = Array(
      (t, identity[IR], identity[Row]),
      (TInt32, GetField(_, "b"), Option(_).map(_.get(1)).orNull),
      (TFloat64, GetField(_, "e"), Option(_).map(_.get(4)).orNull),
      (TBoolean, GetField(_, "f"), Option(_).map(_.get(5)).orNull),
      (TString, GetField(_, "g"), Option(_).map(_.get(6)).orNull),
      (TArray(TInt32), GetField(_, "h"), Option(_).map(_.get(7)).orNull)
    )

    val keyTransformations: Array[(Type, IR => IR)] = Array(
      (TInt32, GetField(_, "b")),
      (TFloat64, GetField(_, "e")),
      (TString, GetField(_, "g")),
      (TStruct("x" -> TInt32, "y" -> TInt64), GetField(_, "a"))
    )

    def test(n: Int, data: IndexedSeq[Row], valueType: Type, valueF: IR => IR, resultF: Row => Any, keyType: Type, keyF: IR => IR): Unit = {

      val aggSig = AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(valueType, keyType))
      val seqOpArgs = Array.tabulate(rows.length) { i =>
        val ref = ArrayRef(Ref("rows", TArray(t)), i)
        FastIndexedSeq[IR](valueF(ref), keyF(ref))
      }

      assertAggEquals(aggSig.singletonContainer,
        FastIndexedSeq(I32(n)),
        seqOpArgs,
        expected = rows.take(n).map(resultF),
        args = FastIndexedSeq(("rows", (TArray(t), data))))
    }

    // test counts and data input orderings
    for (
      n <- FastIndexedSeq(0, 1, 4, 100);
        perm <- permutations
    ) {
      test(n, perm, t, identity[IR], identity[Row], TInt32, GetField(_, "b"))
    }

    // test key and value types
    for (
      (vt, valueF, resultF) <- valueTransformations;
        (kt, keyF) <- keyTransformations
    ) {
      test(4, permutations.last, vt, valueF, resultF, kt, keyF)
    }

    // test stable sort
    test(7, rows, t, identity[IR], identity[Row], TInt64, _ => I64(5L))

    // test GC behavior by passing a large collection
    val rows2 = Array.tabulate(1200)(i => Row(i, i.toString)).toFastIndexedSeq
    val t2 = TStruct("a" -> TInt32, "b" -> TString)
    val aggSig2 = AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(t2, TInt32))
    val seqOpArgs2 = Array.tabulate(rows2.length)(i => FastIndexedSeq[IR](
      ArrayRef(Ref("rows", TArray(t2)), i), GetField(ArrayRef(Ref("rows", TArray(t2)), i), "a")))

    assertAggEquals(aggSig2.singletonContainer,
      FastIndexedSeq(I32(17)),
      seqOpArgs2,
      expected = rows2.take(17),
      args = FastIndexedSeq(("rows", (TArray(t2), rows2.reverse))))

    // test inside of aggregation
    val tr = TableRange(10000, 5)
    val ta = TableAggregate(tr, ApplyAggOp(FastIndexedSeq(19),
      FastIndexedSeq(invoke("str", TString, GetField(Ref("row", tr.typ.rowType), "idx")), I32(9999) - GetField(Ref("row", tr.typ.rowType), "idx")),
      AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(TString, TInt32))))

    assertEvalsTo(ta, (0 until 19).map(i => (9999 - i).toString).toFastIndexedSeq)(ExecStrategy.interpretOnly)
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
      "h" -> TArray(TInt32))

    val rows = FastIndexedSeq(
      Row(Row(11, 11L), 1, 1L, 1f, 1d, true, "one", FastIndexedSeq(1, 1)),
      Row(Row(22, 22L), 2, 2L, 2f, 2d, false, "two", null),
      null,
      Row(Row(33, 33L), 3, 3L, 3f, 3d, null, "three", FastIndexedSeq(3, null)),
      Row(null, null, null, null, null, null, null, FastIndexedSeq()),
      Row(Row(null, 44L), 4, 4L, 4f, 4d, true, "four", null),
      Row(Row(55, null), 5, 5L, 5f, 5d, true, null, null),
      null,
      Row(Row(66, 66L), 6, 6L, 6f, 6d, false, "six", FastIndexedSeq(6, 6, 6, 6)),
      Row(null, null, null, null, null, null, null, null),
      Row(Row(77, 77L), 7, 7L, 7f, 7d, false, "seven", FastIndexedSeq()),
      null,
      null,
      Row(null, null, null, null, null, null, null, null),
      Row(Row(88, 88L), 8, 8L, 8f, 8d, null, "eight", null),
      Row(Row(99, 99L), 9, 9L, 9f, 9d, null, "nine", FastIndexedSeq(null)),
      Row(Row(1010, 1010L), 10, 10L, 10f, 10d, false, "ten", FastIndexedSeq()),
      Row(Row(1111, 1111L), 11, 11L, 11f, 11d, true, "eleven", FastIndexedSeq())
    )

    val aggSig = AggSignature(Take(), FastSeq(TInt32), FastSeq(t))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    FastIndexedSeq(0, 1, 3, 8, 10, 15, 30).foreach { i =>
      assertAggEquals(aggSig.singletonContainer,
        FastIndexedSeq(I32(i)),
        seqOpArgs,
        expected = rows.take(i),
        args = FastIndexedSeq(("rows", (TArray(t), rows))))
    }

    val transformations: IndexedSeq[(IR => IR, Row => Any, Type)] = t.fields.map { f =>
      ((x: IR) => GetField(x, f.name),
        (r: Row) => if (r == null) null else r.get(f.index),
        f.typ)
    }.filter(_._3 == TString)

    transformations.foreach { case (irF, rowF, subT) =>
      val aggSig = AggSignature(Take(), FastSeq(TInt32), FastSeq(subT))
      val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](irF(ArrayRef(Ref("rows", TArray(t)), i))))

      val expected = rows.take(10).map(rowF)
      assertAggEquals(aggSig.singletonContainer,
        FastIndexedSeq(I32(10)),
        seqOpArgs,
        expected = expected,
        args = FastIndexedSeq(("rows", (TArray(t), rows))))
    }
  }

  def seqOpOverArray(aggIdx: Int, a: IR, seqOps: IR => IR, state: AggStateSignature): IR = {
    val idx = Ref(genUID(), TInt32)
    val elt = Ref(genUID(), coerce[TArray](a.typ).elementType)

    Begin(FastIndexedSeq(
      SeqOp(aggIdx, FastIndexedSeq(ArrayLen(a)), state, AggElementsLengthCheck()),
      StreamFor(StreamRange(0, ArrayLen(a), 1), idx.name,
        Let(elt.name, ArrayRef(a, idx),
          SeqOp(aggIdx, FastIndexedSeq(idx, seqOps(elt)), state, AggElements())))))
  }

  @Test def testMin() {
    val aggSig = AggSignature(Min(), FastSeq(), FastSeq(TInt64))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    val seqOpArgsNA = Array.tabulate(8)(i => FastIndexedSeq[IR](NA(TInt64)))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = -2L, args = FastIndexedSeq(("rows", (arrayType, rows))))
    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgsNA, expected = null, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testMax() {
    val aggSig = AggSignature(Max(), FastSeq(), FastSeq(TInt64))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    val seqOpArgsNA = Array.tabulate(8)(i => FastIndexedSeq[IR](NA(TInt64)))

    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgs, expected = 7L, args = FastIndexedSeq(("rows", (arrayType, rows))))
    assertAggEquals(aggSig.singletonContainer, FastIndexedSeq(), seqOpArgsNA, expected = null, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testCollectLongs() {
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    assertAggEquals(collectAggSig(TInt64).singletonContainer, FastIndexedSeq(), seqOpArgs,
      expected = FastIndexedSeq(5L, null, -2L, 7L, null, null),
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testCollectStrs() {
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a")))

    assertAggEquals(collectAggSig(TString).singletonContainer, FastIndexedSeq(), seqOpArgs,
      expected = FastIndexedSeq("abcd", null, null, "abcd", null, "foo"),
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testCollectBig() {
    val seqOpArgs = Array.tabulate(100)(i => FastIndexedSeq(I64(i)))
    assertAggEquals(collectAggSig(TInt64).singletonContainer, FastIndexedSeq(), seqOpArgs,
      expected = Array.tabulate(100)(i => i.toLong).toIndexedSeq,
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testArrayElementsAgg() {
    val aggSigs = FastIndexedSeq(pnnAggSig, countAggSig, sumAggSig).map(AggStateSignature.apply)
    val aeState = AggStateSignature(Map(AggElementsLengthCheck() -> AggSignature(AggElementsLengthCheck(), FastSeq(TVoid), FastSeq(TInt32)),
      AggElements() -> AggSignature(AggElements(), FastSeq(), FastSeq(TInt32, TVoid))), AggElementsLengthCheck(), Some(aggSigs))

    val value = FastIndexedSeq(
      FastIndexedSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastIndexedSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastIndexedSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastIndexedSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastIndexedSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastIndexedSeq(null, null, null, Row("f", 5L)))

    val expected =
      FastIndexedSeq(
        Row(Row("a", 4L), 6L, 10L),
        Row(Row("b", 4L), 6L, 9L),
        Row(Row("c", 4L), 6L, 8L),
        Row(Row("f", 5L), 6L, 10L))

    val init = InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp(0, FastIndexedSeq(), pnnAggSig),
      InitOp(1, FastIndexedSeq(), countAggSig),
      InitOp(2, FastIndexedSeq(), sumAggSig)
    ))), aeState, AggElementsLengthCheck())

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { elt =>
        Begin(FastIndexedSeq(
          SeqOp(0, FastIndexedSeq(elt), pnnAggSig),
          SeqOp(1, FastIndexedSeq(), countAggSig),
          SeqOp(2, FastIndexedSeq(GetField(elt, "b")), sumAggSig)))
      }, aeState)
    }

    assertAggEqualsProcessed(aeState, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, value))), 2, None)
  }

  @Test def testNestedArrayElementsAgg() {
    val aggSigs = FastIndexedSeq(sumAggSig).map(AggStateSignature.apply)
    val aeState1 = AggStateSignature(Map(AggElementsLengthCheck() -> AggSignature(AggElementsLengthCheck(), FastSeq(TVoid), FastSeq(TInt32)),
      AggElements() -> AggSignature(AggElements(), FastSeq(), FastSeq(TInt32, TVoid))), AggElementsLengthCheck(), Some(aggSigs))

    val aggSigs2 = FastIndexedSeq(aeState1)
    val aeState2 = AggStateSignature(Map(AggElementsLengthCheck() -> AggSignature(AggElementsLengthCheck(), FastSeq(TVoid), FastSeq(TInt32)),
      AggElements() -> AggSignature(AggElements(), FastSeq(), FastSeq(TInt32, TVoid))), AggElementsLengthCheck(), Some(aggSigs2))

    val init = InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
        InitOp(0, FastIndexedSeq(), sumAggSig)
      ))), aeState1, AggElementsLengthCheck())
    ))), aeState2, AggElementsLengthCheck())

    val stream = Ref("stream", TArray(TArray(TArray(TInt64))))
    val seq = Array.tabulate(10) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { array1 =>
        seqOpOverArray(0, array1, { elt =>
          SeqOp(0, FastIndexedSeq(elt), sumAggSig)
        }, aeState1)
      }, aeState2)
    }

    val expected = FastIndexedSeq(Row(FastIndexedSeq(Row(45L))))

    val args = Array.tabulate(10)(i => FastIndexedSeq(FastIndexedSeq(i.toLong))).toFastIndexedSeq
    assertAggEqualsProcessed(aeState2, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, args))), 2, None)
  }

  @Test def testArrayElementsAggTake() {
    val value = FastIndexedSeq(
      FastIndexedSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastIndexedSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastIndexedSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastIndexedSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastIndexedSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastIndexedSeq(null, null, null, Row("f", 5L)))

    val take = AggSignature(Take(), FastSeq(TInt32), FastSeq(t))

    val aggSigs = FastIndexedSeq(take).map(AggStateSignature.apply)
    val aeState = AggStateSignature(Map(AggElementsLengthCheck() -> AggSignature(AggElementsLengthCheck(), FastSeq(TVoid), FastSeq(TInt32)),
      AggElements() -> AggSignature(AggElements(), FastSeq(), FastSeq(TInt32, TVoid))), AggElementsLengthCheck(), Some(aggSigs))

    val init = InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp(0, FastIndexedSeq(I32(3)), take)
    ))), aeState, AggElementsLengthCheck())

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { elt =>
        SeqOp(0, FastIndexedSeq(elt), take)
      }, aeState)
    }

    val expected = Array.tabulate(value(0).length)(i => Row(Array.tabulate(3)(j => value(j)(i)).toFastIndexedSeq)).toFastIndexedSeq
    assertAggEqualsProcessed(aeState, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, value))), 2, None)
  }

  @Test def testGroup() {
    val pnn = AggSignature(PrevNonnull(), FastSeq(), FastSeq(t))
    val count = AggSignature(Count(), FastSeq(), FastSeq())
    val sum = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))

    val kt = TString
    val group = AggSignature(Group(), FastSeq(TVoid), FastSeq(kt, TVoid))
    val grouped = AggStateSignature(Map(Group() -> group), Group(), Some(FastSeq(pnn, count, sum).map(_.singletonContainer)))

    val initOpArgs = FastIndexedSeq(Begin(FastIndexedSeq(
      InitOp(0, FastIndexedSeq(), pnn),
      InitOp(1, FastIndexedSeq(), count),
      InitOp(2, FastIndexedSeq(), sum))))

    val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
        Begin(FastIndexedSeq(
          SeqOp(0, FastIndexedSeq(ArrayRef(rref, i)), pnn),
          SeqOp(1, FastIndexedSeq(), count),
          SeqOp(2, FastIndexedSeq(GetField(ArrayRef(rref, i), "b")), sum)))))

    val expected = Map(
      "abcd" -> Row(Row("abcd", 7L), 2L, 12L),
      "foo" -> Row(Row("foo", null), 1L, 0L),
      (null, Row(Row(null, -2L), 3L, -2L)))

    assertAggEquals(grouped, initOpArgs, seqOpArgs, expected = expected, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testNestedGroup() {
    val pnn = AggSignature(PrevNonnull(), FastSeq(), FastSeq(t))
    val count = AggSignature(Count(), FastSeq(), FastSeq())
    val sum = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))

    val kt = TString
    val group = AggSignature(Group(), FastSeq(TVoid), FastSeq(kt, TVoid))
    val grouped1 = AggStateSignature(Map(Group() -> group), Group(), Some(FastSeq(pnn, count, sum).map(_.singletonContainer)))
    val grouped2 = AggStateSignature(Map(Group() -> group), Group(), Some(FastSeq(grouped1)))

    val initOpArgs = FastIndexedSeq(
      InitOp(0, FastIndexedSeq(
        Begin(FastIndexedSeq(
          InitOp(0, FastIndexedSeq(), pnn),
          InitOp(1, FastIndexedSeq(), count),
          InitOp(2, FastIndexedSeq(), sum)))
      ), grouped1, Group()))

    val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
        SeqOp(0, FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
          Begin(FastIndexedSeq(
            SeqOp(0, FastIndexedSeq(ArrayRef(rref, i)), pnn),
            SeqOp(1, FastIndexedSeq(), count),
            SeqOp(2, FastIndexedSeq(GetField(ArrayRef(rref, i), "b")), sum)))
        ), grouped1, Group())))

    val expected = Map(
      "abcd" -> Row(Map("abcd" -> Row(Row("abcd", 7L), 2L, 12L))),
      "foo" -> Row(Map("foo" -> Row(Row("foo", null), 1L, 0L))),
      (null, Row(Map((null, Row(Row(null, -2L), 3L, -2L))))))

    assertAggEquals(grouped2, initOpArgs, seqOpArgs, expected = expected, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testCollectAsSet() {
    val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))
    val elts = Array.tabulate(rows.length)(i => FastIndexedSeq(GetField(ArrayRef(rref, i), "a")))
    val eltsPrimitive = Array.tabulate(rows.length)(i => FastIndexedSeq(GetField(ArrayRef(rref, i), "b")))

    val expected = Set("abcd", "foo", null)
    val expectedPrimitive = Set(5L, -2L, 7L, null)

    val aggsig = AggSignature(CollectAsSet(), FastSeq(), FastSeq(TString))
    val aggsigPrimitive = AggSignature(CollectAsSet(), FastSeq(), FastSeq(TInt64))
    assertAggEquals(aggsig.singletonContainer, FastSeq(), elts, expected = expected, args = FastIndexedSeq(("rows", (arrayType, rows))), expectedInit = Some(Set()))
    assertAggEquals(aggsigPrimitive.singletonContainer, FastSeq(), eltsPrimitive, expected = expectedPrimitive, args = FastIndexedSeq(("rows", (arrayType, rows))), expectedInit = Some(Set()))
  }

  @Test def testDownsample() {
    val aggSig = AggSignature(Downsample(), FastSeq(TInt32), FastSeq(TFloat64, TFloat64, TArray(TString)))
    val rows = FastIndexedSeq(
      Row(-1.23, 1.23, null),
      Row(-10d, 10d, FastIndexedSeq("foo")),
      Row(0d, 100d, FastIndexedSeq()),
      Row(0d, 100d, FastIndexedSeq()),
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
      Row(3d, 3.3d, null)
    )

    val arrayType = TArray(TStruct("x" -> TFloat64, "y" -> TFloat64, "label" -> TArray(TString)))
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](
      GetField(ArrayRef(Ref("rows", arrayType), i), "x"),
      GetField(ArrayRef(Ref("rows", arrayType), i), "y"),
      GetField(ArrayRef(Ref("rows", arrayType), i), "label")
    ))

    assertAggEquals(aggSig.singletonContainer,
      FastIndexedSeq(I32(500)),
      Array.fill[IndexedSeq[IR]](20)(FastIndexedSeq(NA(TFloat64), NA(TFloat64), NA(TArray(TString)))),
      expected = FastIndexedSeq(),
      args = FastIndexedSeq(("rows", (arrayType, rows))))

    val expected = rows.toSet
    assertAggEquals(aggSig.singletonContainer,
      FastIndexedSeq(I32(100)),
      seqOpArgs,
      expected = expected,
      args = FastIndexedSeq(("rows", (arrayType, rows))),
      transformResult = Some(_.asInstanceOf[IndexedSeq[_]].toSet))
  }

  @Test def testLoweringMatrixMapColsWithAggFilterAndLets(): Unit = {
    val t = MatrixType(TStruct.empty, FastIndexedSeq("col_idx"), TStruct("col_idx" -> TInt32), FastIndexedSeq("row_idx"), TStruct("row_idx" -> TInt32), TStruct.empty)
    val ir = TableCollect(MatrixColsTable(MatrixMapCols(
      MatrixRead(t, false, false, MatrixRangeReader(10, 10, None)),
      InsertFields(Ref("sa", t.colType), FastSeq(("foo",
        Let("bar",
          GetField(Ref("sa", t.colType), "col_idx") + I32(1),
          AggFilter(
            GetField(Ref("va", t.rowType), "row_idx") < I32(5),
            Ref("bar", TInt32).toL + Ref("bar", TInt32).toL + ApplyAggOp(
              FastIndexedSeq(),
              FastIndexedSeq(GetField(Ref("va", t.rowType), "row_idx").toL),
              AggSignature(Sum(), FastSeq(), FastSeq(TInt64))),
            false))))),
      Some(FastIndexedSeq()))))
    assertEvalsTo(ir, Row((0 until 10).map(i => Row(i, 2L * i + 12L)), Row()))(ExecStrategy.interpretOnly)
  }

  @Test def testRunAggScan(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = AggSignature(Sum(), FastSeq(), FastSeq(TFloat64))
    val x = ToArray(RunAggScan(
      StreamRange(I32(0), I32(5), I32(1)),
      "foo",
      InitOp(0, FastSeq(), sig),
      SeqOp(0, FastIndexedSeq(Ref("foo", TInt32).toD), sig),
      GetTupleElement(ResultOp(0, Array(sig.singletonContainer)), 0),
      Array(sig.singletonContainer)))
    assertEvalsTo(x, FastIndexedSeq(0.0, 0.0, 1.0, 3.0, 6.0))
  }

  @Test def testNestedRunAggScan(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = AggSignature(Sum(), FastSeq(), FastSeq(TFloat64))
    val x =
      ToArray(
        StreamFlatMap(
          StreamRange(I32(3), I32(6), I32(1)),
          "i",
          RunAggScan(
            StreamRange(I32(0), Ref("i", TInt32), I32(1)),
            "foo",
            InitOp(0, FastSeq(), sig),
            SeqOp(0, FastIndexedSeq(Ref("foo", TInt32).toD), sig),
            GetTupleElement(ResultOp(0, Array(sig.singletonContainer)), 0),
            Array(sig.singletonContainer))))
    assertEvalsTo(x, FastIndexedSeq(
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0, 3.0,
      0.0, 0.0, 1.0, 3.0, 6.0))
  }

  @Test def testRunAggBasic(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sig = AggSignature(Sum(), FastSeq(), FastSeq(TFloat64))
    val x = RunAgg(
      Begin(FastSeq(
        InitOp(0, FastSeq(), sig),
        SeqOp(0, FastSeq(F64(1.0)), sig),
        SeqOp(0, FastSeq(F64(-5.0)), sig))),
      ResultOp(0, FastIndexedSeq(sig.singletonContainer)),
      FastIndexedSeq(sig.singletonContainer))
    assertEvalsTo(x, Row(-4.0))
  }

  @Test def testRunAggNested(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val sumSig = AggSignature(Sum(), FastSeq(), FastSeq(TFloat64))
    val takeSig = AggSignature(Take(), FastSeq(TInt32), FastSeq(TFloat64))
    val x = RunAgg(
      Begin(FastSeq(
        InitOp(0, FastSeq(I32(5)), takeSig),
        StreamFor(
          StreamRange(I32(0), I32(10), I32(1)),
          "foo",
          SeqOp(0, FastSeq(
            RunAgg(
              Begin(FastSeq(
                InitOp(0, FastSeq(), sumSig),
                SeqOp(0, FastSeq(F64(-1.0)), sumSig),
                SeqOp(0, FastSeq(Ref("foo", TInt32).toD), sumSig))),
              GetTupleElement(ResultOp(0, FastIndexedSeq(sumSig.singletonContainer)), 0),
              FastIndexedSeq(sumSig.singletonContainer))
          ), takeSig)
        ))
      ),
      GetTupleElement(ResultOp(0, FastIndexedSeq(takeSig.singletonContainer)), 0),
      FastIndexedSeq(takeSig.singletonContainer))
    assertEvalsTo(x, FastIndexedSeq(-1d, 0d, 1d, 2d, 3d))
  }

  @Test(enabled = false) def testAggStateAndCombOp(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val takeSig = AggSignature(Take(), FastSeq(TInt32), FastSeq(TInt64))
    val x = Let(
      "x",
      RunAgg(
        Begin(FastSeq(
          InitOp(0, FastSeq(I32(10)), takeSig),
          SeqOp(0, FastSeq(NA(TInt64)), takeSig),
          SeqOp(0, FastSeq(I64(-1l)), takeSig),
          SeqOp(0, FastSeq(I64(2l)), takeSig)
        )),
        AggStateValue(0, takeSig.singletonContainer),
        FastIndexedSeq(takeSig.singletonContainer)),
      RunAgg(
        Begin(FastSeq(
          InitOp(0, FastSeq(I32(10)), takeSig),
          CombOpValue(0, Ref("x", TBinary), takeSig.singletonContainer),
          SeqOp(0, FastSeq(I64(3l)), takeSig),
          CombOpValue(0, Ref("x", TBinary), takeSig.singletonContainer),
          SeqOp(0, FastSeq(I64(0l)), takeSig))),
        GetTupleElement(ResultOp(0, FastIndexedSeq(takeSig.singletonContainer)), 0),
        FastIndexedSeq(takeSig.singletonContainer)))

    assertEvalsTo(x, FastIndexedSeq(null, -1l, 2l, 3l, null, null, -1l, 2l, 0l))
  }
}
