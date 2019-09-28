package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.utils._
import is.hail.variant.{Call0, Call1, Call2}
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class Aggregators2Suite extends HailSuite {

  def assertAggEquals(
    aggSig: AggSignature2,
    initOp: IR,
    seqOps: IndexedSeq[IR],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))],
    nPartitions: Int,
    expectedInit: Option[Any]
  ): Unit = {
    assert(seqOps.length >= 2 * nPartitions, s"Test aggregators with a larger stream!")

    val argT = PType.canonical(TStruct(args.map { case (n, (typ, _)) => n -> typ }: _*)).asInstanceOf[PStruct]
    val argVs = Row.fromSeq(args.map { case (_, (_, v)) => v })
    val argRef = Ref(genUID(), argT.virtualType)
    val spec = BufferSpec.defaultUncompressed

    val (_, combAndDuplicate) = CompileWithAggregators2[Unit](
      Array.fill(nPartitions)(aggSig),
      Begin(
        Array.tabulate(nPartitions)(i => DeserializeAggs(i, i, spec, Array(aggSig))) ++
          Array.range(1, nPartitions).map(i => CombOp2(0, i, aggSig)) :+
          SerializeAggs(0, 0, spec, Array(aggSig)) :+
          DeserializeAggs(1, 0, spec, Array(aggSig))))

    val (rt: PTuple, resF) = CompileWithAggregators2[Long](
      Array.fill(nPartitions)(aggSig),
      ResultOp2(0, Array(aggSig, aggSig)))
    assert(rt.types(0) == rt.types(1))

    val resultType = rt.types(0)
    assert(resultType.virtualType.typeCheck(expected), s"expected type $resultType")

    Region.scoped { region =>
      val argOff = ScalaToRegionValue(region, argT, argVs)

      def withArgs(foo: IR) = {
        CompileWithAggregators2[Long, Unit](
          Array(aggSig),
          argRef.name, argRef.pType,
          args.map(_._1).foldLeft[IR](foo) { case (op, name) =>
            Let(name, GetField(argRef, name), op)
          })._2
      }

      val serialize = SerializeAggs(0, 0, spec, Array(aggSig))
      val (_, writeF) = CompileWithAggregators2[Unit](
        Array(aggSig),
        serialize)

      val initF = withArgs(initOp)

      expectedInit.foreach { v =>
        val (rt: PBaseStruct, resOneF) = CompileWithAggregators2[Long](
          Array(aggSig), ResultOp2(0, Array(aggSig)))

        val init = initF(0, region)
        val res = resOneF(0, region)

        Region.smallScoped { aggRegion =>
          init.newAggState(aggRegion)
          init(region, argOff, false)
          res.setAggState(aggRegion, init.getAggOffset())
          val result = SafeRow(rt, region, res(region)).get(0)
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
        val double = SafeRow(rt, region, res(region))
        assert(resultType.virtualType.valuesSimilar(double.get(0), double.get(1)), // state does not change through serialization
          s"\nbefore: ${ double.get(0) }\nafter:  ${ double.get(1) }")
        assert(resultType.virtualType.valuesSimilar(double.get(0), expected),
          s"\nresult: ${ double.get(0) }\nexpect: $expected")
      }
    }
  }

  def assertAggEquals(
    aggSig: AggSignature2,
    initArgs: IndexedSeq[IR],
    seqArgs: IndexedSeq[IndexedSeq[IR]],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastIndexedSeq(),
    nPartitions: Int = 2,
    expectedInit: Option[Any] = None): Unit =
    assertAggEquals(aggSig,
      InitOp2(0, initArgs, aggSig),
      seqArgs.map(s => SeqOp2(0, s, aggSig)),
      expected, args, nPartitions, expectedInit)

  val t = TStruct("a" -> TString(), "b" -> TInt64())
  val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
  val arrayType = TArray(t)

  val pnnAggSig = AggSignature2(PrevNonnull(), FastSeq[Type](), FastSeq[Type](t), None)
  val countAggSig = AggSignature2(Count(), FastSeq[Type](), FastSeq[Type](), None)
  val sumAggSig = AggSignature2(Sum(), FastSeq[Type](), FastSeq[Type](TInt64()), None)
  def collectAggSig(t: Type): AggSignature2 = AggSignature2(Collect(), FastSeq(), FastSeq(t), None)

  @Test def TestCount() {
    val aggSig = AggSignature2(Count(), FastSeq(), FastSeq(), None)
    val seqOpArgs = Array.fill(rows.length)(FastIndexedSeq[IR]())

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = rows.length.toLong, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testSum() {
    val aggSig = AggSignature2(Sum(), FastSeq(), FastSeq(TInt64()), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = 10L, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testPrevNonnullStr() {
    val aggSig = AggSignature2(PrevNonnull(), FastSeq(), FastSeq(TString()), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a")))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = rows.last.get(0), args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testPrevNonnull() {
    val aggSig = AggSignature2(PrevNonnull(), FastSeq(), FastSeq(t), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = rows.last, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testProduct() {
    val aggSig = AggSignature2(Product(), FastSeq(), FastSeq(TInt64()), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = -70L, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testCallStats() {
    val t = TStruct("x" -> TCall())

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

    val aggSig = AggSignature2(CallStats(), FastSeq(TInt32()), FastSeq(TCall()), None)

    def seqOpArgs(calls: IndexedSeq[Any]) = Array.tabulate(calls.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(Ref("calls", TArray(t)), i), "x")))

    val an = 18
    val ac = FastIndexedSeq(10, 6, 1, 1, 0)
    val af = ac.map(_.toDouble / an).toFastIndexedSeq
    val homCount = FastIndexedSeq(3, 1, 0, 0, 0)
    assertAggEquals(aggSig,
      FastIndexedSeq(I32(5)),
      seqOpArgs(calls),
      expected = Row(ac, af, an, homCount),
      args = FastIndexedSeq(("calls", (TArray(t), calls))))

    val allMissing = calls.filter(_ == null)
    assertAggEquals(aggSig,
      FastIndexedSeq(I32(5)),
      seqOpArgs(allMissing),
      expected = Row(FastIndexedSeq(0, 0, 0, 0, 0), null, 0, FastIndexedSeq(0, 0, 0, 0, 0)),
      args = FastIndexedSeq(("calls", (TArray(t), allMissing))))
  }

  @Test def testTakeBy() {
    val t = TStruct(
      "a" -> TStruct("x" -> TInt32(), "y" -> TInt64()),
      "b" -> TInt32(),
      "c" -> TInt64(),
      "d" -> TFloat32(),
      "e" -> TFloat64(),
      "f" -> TBoolean(),
      "g" -> TString(),
      "h" -> TArray(TInt32()))

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
      (TInt32(), GetField(_, "b"), Option(_).map(_.get(1)).orNull),
      (TFloat64(), GetField(_, "e"), Option(_).map(_.get(4)).orNull),
      (TBoolean(), GetField(_, "f"), Option(_).map(_.get(5)).orNull),
      (TString(), GetField(_, "g"), Option(_).map(_.get(6)).orNull),
      (TArray(TInt32()), GetField(_, "h"), Option(_).map(_.get(7)).orNull)
    )

    val keyTransformations: Array[(Type, IR => IR)] = Array(
      (TInt32(), GetField(_, "b")),
      (TFloat64(), GetField(_, "e")),
      (TString(), GetField(_, "g")),
      (TStruct("x" -> TInt32(), "y" -> TInt64()), GetField(_, "a"))
    )

    def test(n: Int, data: IndexedSeq[Row], valueType: Type, valueF: IR => IR, resultF: Row => Any, keyType: Type, keyF: IR => IR): Unit = {

      val aggSig = AggSignature2(TakeBy(), FastSeq(TInt32()), FastSeq(valueType, keyType), None)
      val seqOpArgs = Array.tabulate(rows.length) { i =>
        val ref = ArrayRef(Ref("rows", TArray(t)), i)
        FastIndexedSeq[IR](valueF(ref), keyF(ref))
      }

      assertAggEquals(aggSig,
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
      test(n, perm, t, identity[IR], identity[Row], TInt32(), GetField(_, "b"))
    }

    // test key and value types
    for (
      (vt, valueF, resultF) <- valueTransformations;
      (kt, keyF) <- keyTransformations
    ) {
      test(4, permutations.last, vt, valueF, resultF, kt, keyF)
    }

    // test stable sort
    test(7, rows, t, identity[IR], identity[Row], TInt64(), _ => I64(5L))

    // test GC behavior by passing a large collection
    val rows2 = Array.tabulate(1200)(i => Row(i, i.toString)).toFastIndexedSeq
    val t2 = TStruct("a" -> TInt32(), "b" -> TString())
    val aggSig2 = AggSignature2(TakeBy(), FastSeq(TInt32()), FastSeq(t2, TInt32()), None)
    val seqOpArgs2 = Array.tabulate(rows2.length)(i => FastIndexedSeq[IR](
      ArrayRef(Ref("rows", TArray(t2)), i), GetField(ArrayRef(Ref("rows", TArray(t2)), i), "a")))

    assertAggEquals(aggSig2,
      FastIndexedSeq(I32(17)),
      seqOpArgs2,
      expected = rows2.take(17),
      args = FastIndexedSeq(("rows", (TArray(t2), rows2.reverse))))

    // test inside of aggregation
    val tr = TableRange(10000, 5)
    val ta = TableAggregate(tr, ApplyAggOp(FastIndexedSeq(19),
      None,
      FastIndexedSeq(invoke("str", TString(), GetField(Ref("row", tr.typ.rowType), "idx")), I32(9999) - GetField(Ref("row", tr.typ.rowType), "idx")),
      AggSignature(TakeBy(), FastSeq(TInt32()), None, FastSeq(TString(), TInt32()))))

    assertEvalsTo(ta, (0 until 19).map(i => (9999 - i).toString).toFastIndexedSeq)(ExecStrategy.interpretOnly)
  }

  @Test def testTake() {
    val t = TStruct(
      "a" -> TStruct("x" -> TInt32(), "y" -> TInt64()),
      "b" -> TInt32(),
      "c" -> TInt64(),
      "d" -> TFloat32(),
      "e" -> TFloat64(),
      "f" -> TBoolean(),
      "g" -> TString(),
      "h" -> TArray(TInt32()))

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

    val aggSig = AggSignature2(Take(), FastSeq(TInt32()), FastSeq(t), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    FastIndexedSeq(0, 1, 3, 8, 10, 15, 30).foreach { i =>
      assertAggEquals(aggSig,
        FastIndexedSeq(I32(i)),
        seqOpArgs,
        expected = rows.take(i),
        args = FastIndexedSeq(("rows", (TArray(t), rows))))
    }

    val transformations: IndexedSeq[(IR => IR, Row => Any, Type)] = t.fields.map { f =>
      ((x: IR) => GetField(x, f.name),
        (r: Row) => if (r == null) null else r.get(f.index),
        f.typ)
    }.filter(_._3 == TString())

    transformations.foreach { case (irF, rowF, subT) =>
      val aggSig = AggSignature2(Take(), FastSeq(TInt32()), FastSeq(subT), None)
      val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](irF(ArrayRef(Ref("rows", TArray(t)), i))))

      val expected = rows.take(10).map(rowF)
      assertAggEquals(aggSig,
        FastIndexedSeq(I32(10)),
        seqOpArgs,
        expected = expected,
        args = FastIndexedSeq(("rows", (TArray(t), rows))))
    }
  }

  def seqOpOverArray(aggIdx: Int, a: IR, seqOps: IR => IR, lcSig: AggSignature2): IR = {
    val idx = Ref(genUID(), TInt32())
    val elt = Ref(genUID(), coerce[TArray](a.typ).elementType)

    val eltSig = AggSignature2(AggElements(), FastSeq[Type](), FastSeq[Type](TInt32(), TVoid), lcSig.nested)

    Begin(FastIndexedSeq(
      SeqOp2(aggIdx, FastIndexedSeq(ArrayLen(a)), lcSig),
      ArrayFor(ArrayRange(0, ArrayLen(a), 1), idx.name,
        Let(elt.name, ArrayRef(a, idx),
          SeqOp2(aggIdx, FastIndexedSeq(idx, seqOps(elt)), eltSig)))))
  }

  @Test def testMin() {
    val aggSig = AggSignature2(Min(), FastSeq[Type](), FastSeq[Type](TInt64()), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    val seqOpArgsNA = Array.tabulate(8)(i => FastIndexedSeq[IR](NA(TInt64())))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = -2L, args = FastIndexedSeq(("rows", (arrayType, rows))))
    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgsNA, expected = null, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testMax() {
    val aggSig = AggSignature2(Max(), FastSeq[Type](), FastSeq[Type](TInt64()), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    val seqOpArgsNA = Array.tabulate(8)(i => FastIndexedSeq[IR](NA(TInt64())))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = 7L, args = FastIndexedSeq(("rows", (arrayType, rows))))
    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgsNA, expected = null, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testCollectLongs() {
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "b")))
    assertAggEquals(collectAggSig(TInt64()), FastIndexedSeq(), seqOpArgs,
      expected = FastIndexedSeq(5L, null, -2L, 7L, null, null),
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testCollectStrs() {
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](GetField(ArrayRef(Ref("rows", arrayType), i), "a")))

    assertAggEquals(collectAggSig(TString()), FastIndexedSeq(), seqOpArgs,
      expected = FastIndexedSeq("abcd", null, null, "abcd", null, "foo"),
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testCollectBig() {
    val seqOpArgs = Array.tabulate(100)(i => FastIndexedSeq(I64(i)))
    assertAggEquals(collectAggSig(TInt64()), FastIndexedSeq(), seqOpArgs,
      expected = Array.tabulate(100)(i => i.toLong).toIndexedSeq,
      args = FastIndexedSeq(("rows", (arrayType, rows)))
    )
  }

  @Test def testArrayElementsAgg() {
    val aggSigs = FastIndexedSeq(pnnAggSig, countAggSig, sumAggSig)
    val lcAggSig = AggSignature2(AggElementsLengthCheck(), FastSeq[Type](TVoid), FastSeq[Type](TInt32()), Some(aggSigs))

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

    val init = InitOp2(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp2(0, FastIndexedSeq(), pnnAggSig),
      InitOp2(1, FastIndexedSeq(), countAggSig),
      InitOp2(2, FastIndexedSeq(), sumAggSig)
    ))), lcAggSig)

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { elt =>
        Begin(FastIndexedSeq(
          SeqOp2(0, FastIndexedSeq(elt), pnnAggSig),
          SeqOp2(1, FastIndexedSeq(), countAggSig),
          SeqOp2(2, FastIndexedSeq(GetField(elt, "b")), sumAggSig)))
      }, lcAggSig)
    }

    assertAggEquals(lcAggSig, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, value))), 2, None)
  }

  @Test def testNestedArrayElementsAgg() {
    val lcAggSig1 = AggSignature2(AggElementsLengthCheck(),
      FastSeq[Type](TVoid), FastSeq[Type](TInt32()),
      Some(FastIndexedSeq(sumAggSig)))
    val lcAggSig2 = AggSignature2(AggElementsLengthCheck(),
      FastSeq[Type](TVoid), FastSeq[Type](TInt32()),
      Some(FastIndexedSeq(lcAggSig1)))

    val init = InitOp2(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp2(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
        InitOp2(0, FastIndexedSeq(), sumAggSig)
      ))), lcAggSig1)
    ))), lcAggSig2)

    val stream = Ref("stream", TArray(TArray(TArray(TInt64()))))
    val seq = Array.tabulate(10) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { array1 =>
        seqOpOverArray(0, array1, { elt =>
          SeqOp2(0, FastIndexedSeq(elt), sumAggSig)
        }, lcAggSig1)
      }, lcAggSig2)
    }

    val expected = FastIndexedSeq(Row(FastIndexedSeq(Row(45L))))

    val args = Array.tabulate(10)(i => FastIndexedSeq(FastIndexedSeq(i.toLong))).toFastIndexedSeq
    assertAggEquals(lcAggSig2, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, args))), 2, None)
  }

  @Test def testArrayElementsAggTake() {
    val value = FastIndexedSeq(
      FastIndexedSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastIndexedSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastIndexedSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastIndexedSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastIndexedSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastIndexedSeq(null, null, null, Row("f", 5L)))

    val take = AggSignature2(Take(), FastIndexedSeq(TInt32()), FastIndexedSeq(t), None)

    val lcAggSig = AggSignature2(AggElementsLengthCheck(),
      FastSeq[Type](TVoid), FastSeq[Type](TInt32()),
      Some(FastIndexedSeq(take)))

    val init = InitOp2(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
      InitOp2(0, FastIndexedSeq(I32(3)), take)
    ))), lcAggSig)

    val stream = Ref("stream", TArray(arrayType))
    val seq = Array.tabulate(value.length) { i =>
      seqOpOverArray(0, ArrayRef(stream, i), { elt =>
        SeqOp2(0, FastIndexedSeq(elt), take)
      }, lcAggSig)
    }

    val expected = Array.tabulate(value(0).length)(i => Row(Array.tabulate(3)(j => value(j)(i)).toFastIndexedSeq)).toFastIndexedSeq
    assertAggEquals(lcAggSig, init, seq, expected, FastIndexedSeq(("stream", (stream.typ, value))), 2, None)
  }

  @Test def testGroup() {
    val pnn = AggSignature2(PrevNonnull(), FastSeq(), FastSeq(t), None)
    val count = AggSignature2(Count(), FastSeq(), FastSeq(), None)
    val sum = AggSignature2(Sum(), FastSeq(), FastSeq(TInt64()), None)

    val kt = TString()
    val grouped = AggSignature2(Group(), FastSeq(TVoid), FastSeq(kt, TVoid), Some(FastSeq(pnn, count, sum)))

    val initOpArgs = FastIndexedSeq(Begin(FastIndexedSeq(
      InitOp2(0, FastIndexedSeq(), pnn),
      InitOp2(1, FastIndexedSeq(), count),
      InitOp2(2, FastIndexedSeq(), sum))))

    val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
        Begin(FastIndexedSeq(
          SeqOp2(0, FastIndexedSeq(ArrayRef(rref, i)), pnn),
          SeqOp2(1, FastIndexedSeq(), count),
          SeqOp2(2, FastIndexedSeq(GetField(ArrayRef(rref, i), "b")), sum)))))

    val expected = Map(
      "abcd" -> Row(Row("abcd", 7L), 2L, 12L),
      "foo" -> Row(Row("foo", null), 1L, 0L),
      (null, Row(Row(null, -2L), 3L, -2L)))

    assertAggEquals(grouped, initOpArgs, seqOpArgs, expected = expected, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testNestedGroup() {
    val pnn = AggSignature2(PrevNonnull(), FastSeq(), FastSeq(t), None)
    val count = AggSignature2(Count(), FastSeq(), FastSeq(), None)
    val sum = AggSignature2(Sum(), FastSeq(), FastSeq(TInt64()), None)

    val kt = TString()
    val grouped1 = AggSignature2(Group(), FastSeq(TVoid), FastSeq(kt, TVoid), Some(FastSeq(pnn, count, sum)))
    val grouped2 = AggSignature2(Group(), FastSeq(TVoid), FastSeq(kt, TVoid), Some(FastSeq(grouped1)))

    val initOpArgs = FastIndexedSeq(
      InitOp2(0, FastIndexedSeq(
        Begin(FastIndexedSeq(
          InitOp2(0, FastIndexedSeq(), pnn),
          InitOp2(1, FastIndexedSeq(), count),
          InitOp2(2, FastIndexedSeq(), sum)))
      ), grouped1))

    val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
    val rref = Ref("rows", TArray(t))

    val seqOpArgs = Array.tabulate(rows.length)(i =>
      FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
        SeqOp2(0, FastIndexedSeq[IR](GetField(ArrayRef(rref, i), "a"),
          Begin(FastIndexedSeq(
            SeqOp2(0, FastIndexedSeq(ArrayRef(rref, i)), pnn),
            SeqOp2(1, FastIndexedSeq(), count),
            SeqOp2(2, FastIndexedSeq(GetField(ArrayRef(rref, i), "b")), sum)))
        ), grouped1)))

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

    val aggsig = AggSignature2(CollectAsSet(), FastSeq(), FastSeq(TString()), None)
    val aggsigPrimitive = AggSignature2(CollectAsSet(), FastSeq(), FastSeq(TInt64()), None)
    assertAggEquals(aggsig, FastSeq(), elts, expected = expected, args = FastIndexedSeq(("rows", (arrayType, rows))), expectedInit = Some(Set()))
    assertAggEquals(aggsigPrimitive, FastSeq(), eltsPrimitive, expected = expectedPrimitive, args = FastIndexedSeq(("rows", (arrayType, rows))), expectedInit = Some(Set()))
  }
}
