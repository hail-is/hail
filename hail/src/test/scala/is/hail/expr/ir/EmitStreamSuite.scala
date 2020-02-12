package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue, RegionValueBuilder, SafeRow, ScalaToRegionValue}
import is.hail.asm4s.{AsmFunction1, AsmFunction3, Code, GenericTypeInfo, MaybeGenericTypeInfo, TypeInfo}
import is.hail.asm4s.joinpoint._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.Call2
import is.hail.HailSuite
import is.hail.expr.ir.lowering.LoweringPipeline
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class EmitStreamSuite extends HailSuite {

  private def compileStream[F >: Null : TypeInfo, A](
    streamIR: IR,
    inputTypes: Seq[PType]
  )(call: (F, Region, A) => Long): A => IndexedSeq[Any] = {
    val argTypeInfos = new ArrayBuilder[MaybeGenericTypeInfo[_]]
    argTypeInfos += GenericTypeInfo[Region]()
    inputTypes.foreach { t =>
      argTypeInfos ++= Seq(GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]())
    }
    val fb = new EmitFunctionBuilder[F](argTypeInfos.result(), GenericTypeInfo[Long])
    val mb = fb.apply_method
    val stream = ExecuteContext.scoped { ctx =>
      EmitStream(new Emit(ctx, mb), streamIR, Env.empty, EmitRegion.default(mb), None)
    }
    mb.emit {
      val arrayt = stream
        .toArrayIterator(mb)
        .toEmitTriplet(mb, PArray(stream.elementType))
      Code(arrayt.setup, arrayt.m.mux(0L, arrayt.v))
    }
    val f = fb.resultWithIndex()
    (arg: A) => Region.scoped { r =>
      val off = call(f(0, r), r, arg)
      if (off == 0L)
        null
      else
        SafeRow.read(PArray(stream.elementType), r, off).asInstanceOf[IndexedSeq[Any]]
    }
  }

  private def compileStream(ir: IR, inputType: PType): Any => IndexedSeq[Any] = {
    type F = AsmFunction3[Region, Long, Boolean, Long]
    compileStream[F, Any](ir, Seq(inputType)) { (f: F, r: Region, arg: Any) =>
      if (arg == null)
        f(r, 0L, true)
      else
        f(r, ScalaToRegionValue(r, inputType, arg), false)
    }
  }

  private def compileStreamWithIter(ir: IR, streamType: PStream): Iterator[Any] => IndexedSeq[Any] = {
    type F = AsmFunction3[Region, Iterator[RegionValue], Boolean, Long]
    compileStream[F, Iterator[Any]](ir, Seq(streamType)) { (f: F, r: Region, it: Iterator[Any]) =>
      val rv = RegionValue(r)
      val rvi = new Iterator[RegionValue] {
        def hasNext: Boolean = it.hasNext
        def next(): RegionValue = {
          rv.setOffset(ScalaToRegionValue(r, streamType.elementType, it.next()))
          rv
        }
      }
      f(r, rvi, it == null)
    }
  }

  private def evalStream(ir: IR): IndexedSeq[Any] =
    compileStream[AsmFunction1[Region, Long], Unit](ir, Seq()) { (f, r, _) => f(r) }
      .apply(())

  private def evalStreamLen(streamIR: IR): Option[Int] = {
    val fb = EmitFunctionBuilder[Region, Int]("eval_stream_len")
    val mb = fb.apply_method
    val stream = ExecuteContext.scoped { ctx =>
      EmitStream(new Emit(ctx, mb), streamIR, Env.empty, EmitRegion.default(mb), None)
    }
    fb.emit {
      JoinPoint.CallCC[Code[Int]] { (jb, ret) =>
        val str = stream.stream
        val mb = fb.apply_method
        str.init(mb, jb, ()) {
          case EmitStream.Missing => ret(0)
          case EmitStream.Start(s0) =>
            str.length(s0) match {
              case Some(len) => ret(len)
              case None => ret(-1)
            }
        }
      }
    }
    val f = fb.resultWithIndex()
    Region.scoped { r =>
      val len = f(0, r)(r)
      if(len < 0) None else Some(len)
    }
  }

  @Test def testEmitNA() {
    assert(evalStream(NA(TStream(TInt32()))) == null)
  }

  @Test def testEmitMake() {
    val x = Ref("x", TInt32())
    val typ = TStream(TInt32())
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      MakeStream(Seq[IR](1, 2, NA(TInt32()), 3), typ) -> IndexedSeq(1, 2, null, 3),
      MakeStream(Seq[IR](), typ) -> IndexedSeq(),
      MakeStream(Seq[IR](MakeTuple.ordered(Seq(4, 5))), TStream(TTuple(TInt32(), TInt32()))) ->
        IndexedSeq(Row(4, 5)),
      MakeStream(Seq[IR](Str("hi"), Str("world")), TStream(TString())) ->
        IndexedSeq("hi", "world")
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitRange() {
    val tripleType = PStruct(false, "start" -> PInt32(), "stop" -> PInt32(), "step" -> PInt32())
    val triple = In(0, tripleType.virtualType)
    val range = compileStream(
      StreamRange(GetField(triple, "start"), GetField(triple, "stop"), GetField(triple, "step")),
      tripleType)
    for {
      start <- -2 to 2
      stop <- -2 to 8
      step <- 1 to 3
    } {
      assert(range(Row(start, stop, step)) == Array.range(start, stop, step).toFastIndexedSeq,
        s"($start, $stop, $step)")
    }
    assert(range(Row(null, 10, 1)) == null)
    assert(range(Row(0, null, 1)) == null)
    assert(range(Row(0, 10, null)) == null)
    assert(range(null) == null)
  }

  @Test def testEmitToStream() {
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ToStream(MakeArray(Seq[IR](), TArray(TInt32()))) -> IndexedSeq(),
      ToStream(MakeArray(Seq[IR](1, 2, 3, 4), TArray(TInt32()))) -> IndexedSeq(1, 2, 3, 4),
      ToStream(NA(TArray(TInt32()))) -> null
    )
    for ((ir, v) <- tests) {
      val expectedLen = Some(if(v == null) 0 else v.length)
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == expectedLen, Pretty(ir))
    }
  }

  @Test def testEmitLet() {
    val Seq(start, end, i) = Seq("start", "end", "i").map(Ref(_, TInt32()))
    val ir =
      Let("end", 10,
        ArrayFlatMap(
          Let("start", 3,
            StreamRange(start, end, 1)),
          "i",
          MakeStream(Seq(i, end), TStream(TInt32())))
      )
    assert(evalStream(ir) == (3 until 10).flatMap { i => Seq(i, 10) }.toIndexedSeq, Pretty(ir))
    assert(evalStreamLen(ir).isEmpty, Pretty(ir))
  }

  @Test def testEmitMap() {
    val ten = StreamRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt32())
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ArrayMap(ten, "x", x * 2) -> (0 until 10).map(_ * 2),
      ArrayMap(ten, "x", x.toL) -> (0 until 10).map(_.toLong),
      ArrayMap(ArrayMap(ten, "x", x + 1), "y", y * y) -> (0 until 10).map(i => (i + 1) * (i + 1)),
      ArrayMap(ten, "x", NA(TInt32())) -> IndexedSeq.tabulate(10) { _ => null }
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitFilter() {
    val ten = StreamRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt64())
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ArrayFilter(ten, "x", x cne 5) -> (0 until 10).filter(_ != 5),
      ArrayFilter(ArrayMap(ten, "x", (x * 2).toL), "y", y > 5L) -> (3 until 10).map(x => (x * 2).toLong),
      ArrayFilter(ArrayMap(ten, "x", (x * 2).toL), "y", NA(TInt32())) -> IndexedSeq(),
      ArrayFilter(ArrayMap(ten, "x", NA(TInt32())), "z", True()) -> IndexedSeq.tabulate(10) { _ => null }
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir).isEmpty, Pretty(ir))
    }
  }

  @Test def testEmitFlatMap() {
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt32())
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ArrayFlatMap(StreamRange(0, 6, 1), "x", StreamRange(0, x, 1)) ->
        (0 until 6).flatMap(0 until _),
      ArrayFlatMap(StreamRange(0, 6, 1), "x", StreamRange(0, NA(TInt32()), 1)) ->
        IndexedSeq(),
      ArrayFlatMap(StreamRange(0, NA(TInt32()), 1), "x", StreamRange(0, x, 1)) ->
        null,
      ArrayFlatMap(StreamRange(0, 20, 1), "x",
        ArrayFlatMap(StreamRange(0, x, 1), "y",
          StreamRange(0, (x + y), 1))) ->
        (0 until 20).flatMap { x => (0 until x).flatMap { y => 0 until (x + y) } },
      ArrayFlatMap(ArrayFilter(StreamRange(0, 5, 1), "x", x cne 3),
        "y", MakeStream(Seq(y, y), TStream(TInt32()))) ->
        IndexedSeq(0, 0, 1, 1, 2, 2, 4, 4),
      ArrayFlatMap(StreamRange(0, 4, 1),
        "x", ToStream(MakeArray(Seq[IR](x, x), TArray(TInt32())))) ->
        IndexedSeq(0, 0, 1, 1, 2, 2, 3, 3)
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      if (v != null)
        assert(evalStreamLen(ir) == None, Pretty(ir))
    }
  }

  @Test def testEmitLeftJoinDistinct() {
    val tupTyp = TTuple(TInt32(), TString())
    val Seq(l, r) = Seq("l", "r").map(Ref(_, tupTyp))
    val Seq(i) = Seq("i").map(Ref(_, TInt32()))
    val cmp = ApplyComparisonOp(
      Compare(TInt32()),
      GetTupleElement(l, 0),
      GetTupleElement(r, 0))

    def leftjoin(lstream: IR, rstream: IR): IR =
      ArrayLeftJoinDistinct(lstream, rstream,
        "l", "r", cmp,
        MakeTuple.ordered(Seq(
          GetTupleElement(l, 1),
          GetTupleElement(r, 1))))

    def pairs(xs: Seq[(Int, String)]): IR =
      MakeStream(xs.map { case (a, b) => MakeTuple.ordered(Seq(I32(a), Str(b))) }, TStream(tupTyp))

    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      leftjoin(pairs(Seq()), pairs(Seq())) -> IndexedSeq(),
      leftjoin(pairs(Seq(3 -> "A")), pairs(Seq())) ->
        IndexedSeq(Row("A", null)),
      leftjoin(pairs(Seq()), pairs(Seq(3 -> "B"))) ->
        IndexedSeq(),
      leftjoin(pairs(Seq(0 -> "A")), pairs(Seq(0 -> "B"))) ->
        IndexedSeq(Row("A", "B")),
      leftjoin(
        pairs(Seq(0 -> "A", 2 -> "B", 3 -> "C")),
        pairs(Seq(0 -> "a", 1 -> ".", 2 -> "b", 4 -> ".."))
      ) -> IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", null)),
      leftjoin(
        pairs(Seq(0 -> "A", 1 -> "B1", 1 -> "B2")),
        pairs(Seq(0 -> "a", 1 -> "b", 2 -> "c"))
      ) -> IndexedSeq(Row("A", "a"), Row("B1", "b"), Row("B2", "b"))
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitScan() {
    val Seq(a, v, x) = Seq("a", "v", "x").map(Ref(_, TInt32()))
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ArrayScan(MakeStream(Seq(), TStream(TInt32())),
        9, "a", "v", a + v) -> IndexedSeq(9),
      ArrayScan(ArrayMap(StreamRange(0, 4, 1), "x", x * x),
        1, "a", "v", a + v) -> IndexedSeq(1, 1/*1+0*0*/, 2/*1+1*1*/, 6/*2+2*2*/, 15/*6+3*3*/)
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitAggScan() {
    def assertAggScan(ir: IR, inType: Type, tests: (Any, Any)*) = {
      val aggregate = compileStream(LoweringPipeline.compileLowerer.apply(ctx, ir, false).asInstanceOf[IR],
        PType.canonical(inType))
      for ((inp, expected) <- tests)
        assert(aggregate(inp) == expected, Pretty(ir))
    }

    def scanOp(op: AggOp, initArgs: Seq[IR], opArgs: Seq[IR]): ApplyScanOp =
      ApplyScanOp(
        initArgs.toFastIndexedSeq,
        opArgs.toFastIndexedSeq,
        AggSignature(op,
          initArgs.map(_.typ),
          opArgs.map(_.typ)))

    val pairType = TStruct("x" -> TCall(), "y" -> TInt32())
    val intsType = TArray(TInt32())

    assertAggScan(
      ArrayAggScan(ToStream(In(0, TArray(pairType))),
        "foo",
        GetField(Ref("foo", pairType), "y") +
          GetField(
            scanOp(CallStats(),
              Seq(I32(2)),
              Seq(GetField(Ref("foo", pairType), "x"))
            ),
            "AN")
      ),
      TArray(pairType),
      FastIndexedSeq(
        Row(null, 1), Row(Call2(0, 0), 2), Row(Call2(0, 1), 3), Row(Call2(1, 1), 4), null, Row(null, 5)
      ) -> FastIndexedSeq(1 + 0, 2 + 0, 3 + 2, 4 + 4, null, 5 + 6)
    )

    assertAggScan(
      ArrayAggScan(
        ArrayAggScan(ToStream(In(0, intsType)),
          "i",
          scanOp(Sum(), Seq(), Seq(Ref("i", TInt32()).toL))),
        "x",
        scanOp(Max(), Seq(), Seq(Ref("x", TInt64())))
      ),
      intsType,
      FastIndexedSeq(2, 5, 8, -3, 2, 2, 1, 0, 0) ->
        IndexedSeq(null, 0L, 2L, 7L, 15L, 15L, 15L, 16L, 17L)
    )
  }

  @Test def testEmitFromIterator() {
    val intsPType = PStream(PInt32Required)

    val f1 = compileStreamWithIter(
      ArrayScan(In(0, intsPType),
        zero = 0,
        "a", "x", Ref("a", TInt32()) + Ref("x", TInt32()) * Ref("x", TInt32())
      ), intsPType)
    assert(f1((1 to 4).iterator) == IndexedSeq(0, 1, 1+4, 1+4+9, 1+4+9+16))
    assert(f1(Iterator.empty) == IndexedSeq(0))
    assert(f1(null) == null)

    val f2 = compileStreamWithIter(
      ArrayFlatMap(
        In(0, intsPType),
        "n", StreamRange(0, Ref("n", TInt32()), 1)
      ), intsPType)
    assert(f2(Seq(1, 5, 2, 9).iterator) == IndexedSeq(1, 5, 2, 9).flatMap(0 until _))
    assert(f2(null) == null)
  }

  @Test def testEmitIf() {
    val xs = MakeStream(Seq[IR](5, 3, 6), TStream(TInt32()))
    val ys = StreamRange(0, 4, 1)
    val na = NA(TStream(TInt32()))
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      If(True(), xs, ys) -> IndexedSeq(5, 3, 6),
      If(False(), xs, ys) -> IndexedSeq(0, 1, 2, 3),
      If(True(), xs, na) -> IndexedSeq(5, 3, 6),
      If(False(), xs, na) -> null,
      If(NA(TBoolean()), xs, ys) -> null,
      ArrayFlatMap(MakeStream(Seq(False(), True(), False()), TStream(TBoolean())),
        "x", If(Ref("x", TBoolean()), xs, ys)) -> IndexedSeq(0, 1, 2, 3, 5, 3, 6, 0, 1, 2, 3)
    )
    val lens: Array[Option[Int]] = Array(Some(3), Some(4), Some(3), Some(0), Some(0), None)
    for (((ir, v), len) <- tests zip lens) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == len, Pretty(ir))
    }
  }

  @Test def testZipIfNA() {

    val t = PCanonicalStruct("missingParam" -> PCanonicalArray(PFloat64()),
      "xs" -> PCanonicalArray(PFloat64()),
      "ys" -> PCanonicalArray(PFloat64()))
    val i1 = Ref("in", t.virtualType)
    val ir = MakeTuple.ordered(Seq(ArrayFold(
      ArrayZip(
        FastIndexedSeq(
          If(IsNA(GetField(i1, "missingParam")), NA(TArray(TFloat64())), GetField(i1, "xs")),
          GetField(i1, "ys")
        ),
        FastIndexedSeq("zipL", "zipR"),
        Ref("zipL", TFloat64()) * Ref("zipR", TFloat64()),
        ArrayZipBehavior.AssertSameLength
      ),
      F64(0d),
      "foldAcc", "foldVal",
      Ref("foldAcc", TFloat64()) + Ref("foldVal", TFloat64())
    )))

    val (pt, f) = Compile[Long, Long](ctx, "in", t, ir)

    Region.smallScoped { r =>
      val rvb = new RegionValueBuilder(r)
      rvb.start(t)
      rvb.addAnnotation(t.virtualType, Row(null, IndexedSeq(1d, 2d), IndexedSeq(3d, 4d)))
      val input = rvb.end()

      assert(SafeRow.read(pt, r, f(0, r)(r, input, false)) == Row(null))
    }
  }
}
