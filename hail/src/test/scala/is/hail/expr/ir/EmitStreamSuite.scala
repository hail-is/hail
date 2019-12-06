package is.hail.expr.ir

import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.asm4s.joinpoint._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.Call2
import is.hail.HailSuite

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class EmitStreamSuite extends HailSuite {

  private def compileStream(streamIR: IR, inputPType: PType): Any => IndexedSeq[Any] = {
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long]("eval_stream")
    val mb = fb.apply_method
    val stream = ExecuteContext.scoped { ctx =>
      EmitStream(new Emit(ctx, mb, 1), streamIR, Env.empty, None, EmitRegion.default(mb), None)
    }
    val eltPType = stream.elementType
    fb.emit {
      val ait = stream.toArrayIterator(mb)
      val arrayt = ait.toEmitTriplet(mb, PArray(eltPType))
      Code(arrayt.setup, arrayt.m.mux(0L, arrayt.v))
    }
    val f = fb.resultWithIndex()
    ({ arg: Any => Region.scoped { r =>
      val off =
        if(arg == null)
          f(0, r)(r, 0L, true)
        else
          f(0, r)(r, ScalaToRegionValue(r, inputPType, arg), false)
      if(off == 0L)
        null
      else
        SafeRow.read(PArray(eltPType), r, off).asInstanceOf[IndexedSeq[Any]]
    } })
  }

  private def evalStream(streamIR: IR): IndexedSeq[Any] =
    compileStream(streamIR, PStruct.empty())(null)

  private def evalStreamLen(streamIR: IR): Option[Int] = {
    val fb = EmitFunctionBuilder[Region, Int]("eval_stream_len")
    val mb = fb.apply_method
    val stream = ExecuteContext.scoped { ctx =>
      EmitStream(new Emit(ctx, mb, 1), streamIR, Env.empty, None, EmitRegion.default(mb), None)
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
      val aggregate = compileStream(ir, inType.physicalType)
      for ((inp, expected) <- tests)
        assert(aggregate(inp) == expected, Pretty(ir))
    }

    def scanOp(op: AggOp, initArgs: Option[Seq[IR]], opArgs: Seq[IR]): ApplyScanOp =
      ApplyScanOp(
        FastIndexedSeq(),
        initArgs.map(_.toFastIndexedSeq),
        opArgs.toFastIndexedSeq,
        AggSignature(op,
          Seq(),
          initArgs.map(_.map(_.typ)),
          opArgs.map(_.typ)))

    val pairType = TStruct("x" -> TCall(), "y" -> TInt32())
    val intsType = TArray(TInt32())

    assertAggScan(
      ArrayAggScan(ToStream(In(0, TArray(pairType))),
        "foo",
        GetField(Ref("foo", pairType), "y") +
          GetField(
            scanOp(CallStats(),
              Some(Seq(I32(2))),
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
          scanOp(Sum(), None, Seq(Ref("i", TInt32()).toL))),
        "x",
        scanOp(Max(), None, Seq(Ref("x", TInt64())))
      ),
      intsType,
      FastIndexedSeq(2, 5, 8, -3, 2, 2, 1, 0, 0) ->
        IndexedSeq(null, 0L, 2L, 7L, 15L, 15L, 15L, 16L, 17L)
    )
  }
}
