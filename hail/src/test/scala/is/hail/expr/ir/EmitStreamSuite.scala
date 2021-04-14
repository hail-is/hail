package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue, RegionValueBuilder, SafeRow, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.Call2
import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.ir.streams.{EmitStream, StreamArgType, StreamUtils}
import is.hail.types.physical.stypes.interfaces.SStreamCode
import org.apache.spark.sql.Row
import is.hail.TestUtils._
import org.testng.annotations.Test

class EmitStreamSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.compileOnly

  private def compile1[T: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Value[T]) => Code[R]): T => R = {
    val fb = EmitFunctionBuilder[T, R](ctx, "stream_test")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getCodeParam[T](1)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  private def compile2[T: TypeInfo, U: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Code[T], Code[U]) => Code[R]): (T, U) => R = {
    val fb = EmitFunctionBuilder[T, U, R](ctx, "F")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getCodeParam[T](1), mb.getCodeParam[U](2)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  private def compile3[T: TypeInfo, U: TypeInfo, V: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Code[T], Code[U], Code[V]) => Code[R]): (T, U, V) => R = {
    val fb = EmitFunctionBuilder[T, U, V, R](ctx, "F")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getCodeParam[T](1), mb.getCodeParam[U](2), mb.getCodeParam[V](3)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  def log(str: Code[String], enabled: Boolean = false): Code[Unit] =
    if (enabled) Code._println(str) else Code._empty

  private def compileStream[F: TypeInfo, T](
    streamIR: IR,
    inputTypes: IndexedSeq[EmitParamType]
  )(call: (F, Region, T) => Long): T => IndexedSeq[Any] = {
    val fb = EmitFunctionBuilder[F](ctx, "F", (classInfo[Region]: ParamType) +: inputTypes.map(pt => pt: ParamType), LongInfo)
    val mb = fb.apply_method
    val ir = streamIR.deepCopy()
    val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
    val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
    InferPType(ir, Env.empty, requiredness, usesAndDefs)

    val emitContext = new EmitContext(ctx, requiredness)

    var arrayType: PType = null
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = mb.getCodeParam[Region](1)
      val s = ir match {
        case ToArray(s) => s
        case s => s
      }
      TypeCheck(s)
      EmitStream.produce(new Emit(emitContext, fb.ecb), s, cb, region, Env.empty, None)
        .consumeCode[Long](cb, 0L, { s =>
          val arr = StreamUtils.toArray(cb, s.asStream.producer, region)
          val scp = SingleCodePCode.fromPCode(cb, arr, region, false)
          arrayType = scp.typ.asInstanceOf[PTypeReferenceSingleCodeType].pt

          coerce[Long](scp.code)
        })
    })
    val f = fb.resultWithIndex()
    (arg: T) =>
      pool.scopedRegion { r =>
        val off = call(f(0, r), r, arg)
        if (off == 0L)
          null
        else
          SafeRow.read(arrayType, off).asInstanceOf[IndexedSeq[Any]]
      }
  }

  private def compileStream(ir: IR, inputType: PType): Any => IndexedSeq[Any] = {
    type F = AsmFunction3RegionLongBooleanLong
    compileStream[F, Any](ir, FastIndexedSeq(SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(inputType)))) { (f: F, r: Region, arg: Any) =>
      if (arg == null)
        f(r, 0L, true)
      else
        f(r, ScalaToRegionValue(r, inputType, arg), false)
    }
  }

  private def compileStreamWithIter(ir: IR, requiresMemoryManagementPerElement: Boolean, streamType: PStream): Iterator[Any] => IndexedSeq[Any] = {
    trait F {
      def apply(o: Region, a: StreamArgType): Long
    }
    compileStream[F, Iterator[Any]](ir,
      IndexedSeq(SingleCodeEmitParamType(true, StreamSingleCodeType(requiresMemoryManagementPerElement, streamType.elementType)))) { (f: F, r: Region, it: Iterator[Any]) =>
      val rvi = new StreamArgType {
        def apply(outerRegion: Region, eltRegion: Region): Iterator[java.lang.Long] =
          new Iterator[java.lang.Long] {
            def hasNext: Boolean = it.hasNext
            def next(): java.lang.Long = {
              ScalaToRegionValue(eltRegion, streamType.elementType, it.next())
            }
          }
      }
      assert(it != null, "null iterators not supported")
      f(r, rvi)
    }
  }

  private def evalStream(ir: IR): IndexedSeq[Any] =
    compileStream[AsmFunction1RegionLong, Unit](ir, FastIndexedSeq()) { (f, r, _) => f(r) }
      .apply(())

  private def evalStreamLen(streamIR: IR): Option[Int] = {
    val fb = EmitFunctionBuilder[Region, Int](ctx, "eval_stream_len")
    val mb = fb.apply_method
    val region = mb.getCodeParam[Region](1)
    val ir = streamIR.deepCopy()
    val usesAndDefs = ComputeUsesAndDefs(ir, errorIfFreeVariables = false)
    val requiredness = Requiredness.apply(ir, usesAndDefs, null, Env.empty) // Value IR inference doesn't need context
    InferPType(ir, Env.empty, requiredness, usesAndDefs)

    val emitContext = new EmitContext(ctx, requiredness)

    fb.emitWithBuilder { cb =>
      TypeCheck(ir)
      val len = cb.newLocal[Int]("len", 0)
      val len2 = cb.newLocal[Int]("len2", -1)

      EmitStream.produce(new Emit(emitContext, fb.ecb), ir, cb, region, Env.empty, None)
        .consume(cb,
          {},
        { case stream: SStreamCode =>
          stream.producer.memoryManagedConsume(region, cb, { cb => stream.producer.length.foreach(c => cb.assign(len2, c))}) { cb =>
            cb.assign(len, len + 1)
          }
        })
      cb.ifx(len2.cne(-1) && (len2.cne(len)),
        cb._fatal(s"length mismatch between computed and iteration length: computed=", len2.toS, ", iter=", len.toS))

      len2
    }
    val f = fb.resultWithIndex()
    pool.scopedRegion { r =>
      val len = f(0, r)(r)
      if (len < 0) None else Some(len)
    }
  }

  @Test def testEmitNA() {
    assert(evalStream(NA(TStream(TInt32))) == null)
  }

  @Test def testEmitMake() {
    val typ = TStream(TInt32)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      MakeStream(Seq[IR](1, 2, NA(TInt32), 3), typ) -> IndexedSeq(1, 2, null, 3),
      MakeStream(Seq[IR](), typ) -> IndexedSeq(),
      MakeStream(Seq[IR](MakeTuple.ordered(Seq(4, 5))), TStream(TTuple(TInt32, TInt32))) ->
        IndexedSeq(Row(4, 5)),
      MakeStream(Seq[IR](Str("hi"), Str("world")), TStream(TString)) ->
        IndexedSeq("hi", "world")
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitRange() {
    val tripleType = PCanonicalStruct(false, "start" -> PInt32(), "stop" -> PInt32(), "step" -> PInt32())
    val range = compileStream(
      StreamRange(
        GetField(In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))), "start"),
        GetField(In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))), "stop"),
        GetField(In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))), "step")),
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
      ToStream(MakeArray(Seq[IR](), TArray(TInt32))) -> IndexedSeq(),
      ToStream(MakeArray(Seq[IR](1, 2, 3, 4), TArray(TInt32))) -> IndexedSeq(1, 2, 3, 4),
      ToStream(NA(TArray(TInt32))) -> null
    )
    for ((ir, v) <- tests) {
      val expectedLen = Option(v).map(_.length)
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == expectedLen, Pretty(ir))
    }
  }

  @Test def testEmitLet() {
    val ir =
      Let("end", 10,
        StreamFlatMap(
          Let("start", 3,
            StreamRange(Ref("start", TInt32), Ref("end", TInt32), 1)),
          "i",
          MakeStream(Seq(Ref("i", TInt32), Ref("end", TInt32)), TStream(TInt32)))
      )
    assert(evalStream(ir) == (3 until 10).flatMap { i => Seq(i, 10) }, Pretty(ir))
    assert(evalStreamLen(ir).isEmpty, Pretty(ir))
  }

  @Test def testEmitMap() {
    def ten = StreamRange(I32(0), I32(10), I32(1))
    def x = Ref("x", TInt32)
    def y = Ref("y", TInt32)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      StreamMap(ten, "x", x * 2) -> (0 until 10).map(_ * 2),
      StreamMap(ten, "x", x.toL) -> (0 until 10).map(_.toLong),
      StreamMap(StreamMap(ten, "x", x + 1), "y", y * y) -> (0 until 10).map(i => (i + 1) * (i + 1)),
      StreamMap(ten, "x", NA(TInt32)) -> IndexedSeq.tabulate(10) { _ => null }
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitFilter() {
    def ten = StreamRange(I32(0), I32(10), I32(1))
    def x = Ref("x", TInt32)
    def y = Ref("y", TInt64)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      StreamFilter(ten, "x", x cne 5) -> (0 until 10).filter(_ != 5),
      StreamFilter(StreamMap(ten, "x", (x * 2).toL), "y", y > 5L) -> (3 until 10).map(x => (x * 2).toLong),
      StreamFilter(StreamMap(ten, "x", (x * 2).toL), "y", NA(TBoolean)) -> IndexedSeq(),
      StreamFilter(StreamMap(ten, "x", NA(TInt32)), "z", True()) -> IndexedSeq.tabulate(10) { _ => null }
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir).isEmpty, Pretty(ir))
    }
  }

  @Test def testEmitFlatMap() {
    def x = Ref("x", TInt32)
    def y = Ref("y", TInt32)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      StreamFlatMap(StreamRange(0, 6, 1), "x", StreamRange(0, x, 1)) ->
        (0 until 6).flatMap(0 until _),
      StreamFlatMap(StreamRange(0, 6, 1), "x", StreamRange(0, NA(TInt32), 1)) ->
        IndexedSeq(),
      StreamFlatMap(StreamRange(0, NA(TInt32), 1), "x", StreamRange(0, x, 1)) ->
        null,
      StreamFlatMap(StreamRange(0, 20, 1), "x",
        StreamFlatMap(StreamRange(0, x, 1), "y",
          StreamRange(0, (x + y), 1))) ->
        (0 until 20).flatMap { x => (0 until x).flatMap { y => 0 until (x + y) } },
      StreamFlatMap(StreamFilter(StreamRange(0, 5, 1), "x", x cne 3),
        "y", MakeStream(Seq(y, y), TStream(TInt32))) ->
        IndexedSeq(0, 0, 1, 1, 2, 2, 4, 4),
      StreamFlatMap(StreamRange(0, 4, 1),
        "x", ToStream(MakeArray(Seq[IR](x, x), TArray(TInt32)))) ->
        IndexedSeq(0, 0, 1, 1, 2, 2, 3, 3)
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      if (v != null)
        assert(evalStreamLen(ir) == None, Pretty(ir))
    }
  }

  @Test def testEmitJoinRightDistinct() {
    val eltType = TStruct("k" -> TInt32, "v" -> TString)

    def join(lstream: IR, rstream: IR, joinType: String): IR =
      StreamJoinRightDistinct(
        lstream, rstream, FastIndexedSeq("k"), FastIndexedSeq("k"), "l", "r",
        MakeTuple.ordered(Seq(
          GetField(Ref("l", eltType), "v"),
          GetField(Ref("r", eltType), "v"))),
        joinType)
    def leftjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "left")
    def outerjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "outer")

    def pairs(xs: Seq[(Int, String)]): IR =
      MakeStream(xs.map { case (a, b) => MakeStruct(Seq("k" -> I32(a), "v" -> Str(b))) }, TStream(eltType))

    val tests: Array[(IR, IR, IndexedSeq[Any], IndexedSeq[Any])] = Array(
      (pairs(Seq()), pairs(Seq()), IndexedSeq(), IndexedSeq()),
      (pairs(Seq(3 -> "A")),
        pairs(Seq()),
        IndexedSeq(Row("A", null)),
        IndexedSeq(Row("A", null)))
//      (pairs(Seq()),
//        pairs(Seq(3 -> "B")),
//        IndexedSeq(),
//        IndexedSeq(Row(null, "B"))),
//      (pairs(Seq(0 -> "A")),
//        pairs(Seq(0 -> "B")),
//        IndexedSeq(Row("A", "B")),
//        IndexedSeq(Row("A", "B"))),
//      (pairs(Seq(0 -> "A", 2 -> "B", 3 -> "C")),
//        pairs(Seq(0 -> "a", 1 -> ".", 2 -> "b", 4 -> "..")),
//        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", null)),
//        IndexedSeq(Row("A", "a"), Row(null, "."), Row("B", "b"), Row("C", null), Row(null, ".."))),
//      (pairs(Seq(0 -> "A", 1 -> "B1", 1 -> "B2")),
//        pairs(Seq(0 -> "a", 1 -> "b", 2 -> "c")),
//        IndexedSeq(Row("A", "a"), Row("B1", "b"), Row("B2", "b")),
//        IndexedSeq(Row("A", "a"), Row("B1", "b"), Row("B2", "b"), Row(null, "c")))
    )
    for ((lstream, rstream, expectedLeft, expectedOuter) <- tests) {
      val l = leftjoin(lstream, rstream)
      val o = outerjoin(lstream, rstream)
      assert(evalStream(l) == expectedLeft, Pretty(l))
      assert(evalStream(o) == expectedOuter, Pretty(o))
      assert(evalStreamLen(l) == Some(expectedLeft.length), Pretty(l))
      assert(evalStreamLen(o) == None, Pretty(o))
    }
  }

  @Test def testEmitScan() {
    def a = Ref("a", TInt32)
    def v = Ref("v", TInt32)
    def x = Ref("x", TInt32)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      StreamScan(MakeStream(Seq(), TStream(TInt32)),
        9, "a", "v", a + v) -> IndexedSeq(9),
      StreamScan(StreamMap(StreamRange(0, 4, 1), "x", x * x),
        1, "a", "v", a + v) -> IndexedSeq(1, 1/*1+0*0*/, 2/*1+1*1*/, 6/*2+2*2*/, 15/*6+3*3*/)
    )
    for ((ir, v) <- tests) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == Some(v.length), Pretty(ir))
    }
  }

  @Test def testEmitAggScan() {
    def assertAggScan(ir: IR, inType: Type, tests: (Any, Any)*): Unit = {
      val aggregate = compileStream(LoweringPipeline.compileLowerer(false).apply(ctx, ir).asInstanceOf[IR],
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

    val pairType = TStruct("x" -> TCall, "y" -> TInt32)
    val intsType = TArray(TInt32)

    assertAggScan(
      StreamAggScan(ToStream(In(0, TArray(pairType))),
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
      StreamAggScan(
        StreamAggScan(ToStream(In(0, intsType)),
          "i",
          scanOp(Sum(), Seq(), Seq(Ref("i", TInt32).toL))),
        "x",
        scanOp(Max(), Seq(), Seq(Ref("x", TInt64)))
      ),
      intsType,
      FastIndexedSeq(2, 5, 8, -3, 2, 2, 1, 0, 0) ->
        IndexedSeq(null, 0L, 2L, 7L, 15L, 15L, 15L, 16L, 17L)
    )
  }

  @Test def testEmitFromIterator() {
    val intsPType = PCanonicalStream(PInt32())

    val f1 = compileStreamWithIter(
      StreamScan(In(0, SingleCodeEmitParamType(true, StreamSingleCodeType(false, PInt32()))),
        zero = 0,
        "a", "x", Ref("a", TInt32) + Ref("x", TInt32) * Ref("x", TInt32)
      ), false, intsPType)
    assert(f1((1 to 4).iterator) == IndexedSeq(0, 1, 1+4, 1+4+9, 1+4+9+16))
    assert(f1(Iterator.empty) == IndexedSeq(0))

    val f2 = compileStreamWithIter(
      StreamFlatMap(
        In(0, SingleCodeEmitParamType(true, StreamSingleCodeType(false, PInt32()))),
        "n", StreamRange(0, Ref("n", TInt32), 1)
      ), false, intsPType)
    assert(f2(Seq(1, 5, 2, 9).iterator) == IndexedSeq(1, 5, 2, 9).flatMap(0 until _))

    val f3 = compileStreamWithIter(
      StreamRange(0, StreamLen(In(0, SingleCodeEmitParamType(true, StreamSingleCodeType(false, PInt32())))), 1), false, intsPType)
    assert(f3(Seq(1, 5, 2, 9).iterator) == IndexedSeq(0, 1, 2, 3))
    assert(f3(Seq().iterator) == IndexedSeq())
  }

  @Test def testEmitIf() {
    def xs = MakeStream(Seq[IR](5, 3, 6), TStream(TInt32))
    def ys = StreamRange(0, 4, 1)
    def na = NA(TStream(TInt32))
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      If(True(), xs, ys) -> IndexedSeq(5, 3, 6),
      If(False(), xs, ys) -> IndexedSeq(0, 1, 2, 3),
      If(True(), xs, na) -> IndexedSeq(5, 3, 6),
      If(False(), xs, na) -> null,
      If(NA(TBoolean), xs, ys) -> null,
      StreamFlatMap(
        MakeStream(Seq(False(), True(), False()), TStream(TBoolean)),
        "x",
        If(Ref("x", TBoolean), xs, ys))
        -> IndexedSeq(0, 1, 2, 3, 5, 3, 6, 0, 1, 2, 3)
    )
    val lens: Array[Option[Int]] = Array(Some(3), Some(4), Some(3), None, None, None)
    for (((ir, v), len) <- tests zip lens) {
      assert(evalStream(ir) == v, Pretty(ir))
      assert(evalStreamLen(ir) == len, Pretty(ir))
    }
  }

  @Test def testZipIfNA() {

    val t = PCanonicalStruct(true, "missingParam" -> PCanonicalArray(PFloat64()),
      "xs" -> PCanonicalArray(PFloat64()),
      "ys" -> PCanonicalArray(PFloat64()))
    val i1 = Ref("in", t.virtualType)
    val ir = MakeTuple.ordered(Seq(StreamFold(
      StreamZip(
        FastIndexedSeq(
          ToStream(If(IsNA(GetField(i1, "missingParam")), NA(TArray(TFloat64)), GetField(i1, "xs"))),
          ToStream(GetField(i1, "ys"))
        ),
        FastIndexedSeq("zipL", "zipR"),
        Ref("zipL", TFloat64) * Ref("zipR", TFloat64),
        ArrayZipBehavior.AssertSameLength
      ),
      F64(0d),
      "foldAcc", "foldVal",
      Ref("foldAcc", TFloat64) + Ref("foldVal", TFloat64)
    )))

    val (Some(PTypeReferenceSingleCodeType(pt)), f) = Compile[AsmFunction2RegionLongLong](ctx,
      FastIndexedSeq(("in", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
      ir)

    pool.scopedSmallRegion { r =>
      val input = t.unstagedStoreJavaObject(Row(null, IndexedSeq(1d, 2d), IndexedSeq(3d, 4d)), r)

      assert(SafeRow.read(pt, f(0, r)(r, input)) == Row(null))
    }
  }

  @Test def testFold() {
    val ints = Literal(TArray(TInt32), IndexedSeq(1, 2, 3, 4))
    val strsLit = Literal(TArray(TString), IndexedSeq("one", "two", "three", "four"))
    val strs = MakeStream(FastIndexedSeq(Str("one"), Str("two"), Str("three"), Str("four")), TStream(TString), true)

    assertEvalsTo(
      foldIR(ToStream(ints, requiresMemoryManagementPerElement = false), I32(-1)) { (acc, elt) => acc + elt },
      9
    )

    assertEvalsTo(
      foldIR(ToStream(strsLit, requiresMemoryManagementPerElement = false), Str("")) { (acc, elt) => invoke("concat", TString, acc, elt)},
      "onetwothreefour"
    )

    assertEvalsTo(
      foldIR(strs, Str("")) { (acc, elt) => invoke("concat", TString, acc, elt)},
      "onetwothreefour"
    )
  }

  @Test def testGrouped() {
    // empty => empty
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 0, 1, false),
            I32(5)
          )) { inner => ToArray(inner) }
      ),
      IndexedSeq())

    // general case where stream ends in inner group
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 10, 1, false),
            I32(3)
          )) { inner => ToArray(inner) }
      ),
      IndexedSeq(
        IndexedSeq(0, 1, 2),
        IndexedSeq(3, 4, 5),
        IndexedSeq(6, 7, 8),
        IndexedSeq(9)))

    // stream ends at end of inner group
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 10, 1, false),
            I32(5)
          )) { inner => ToArray(inner) }
      ),
      IndexedSeq(
        IndexedSeq(0, 1, 2, 3, 4),
        IndexedSeq(5, 6, 7, 8, 9)))

    // separate regions
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            MakeStream((0 until 10).map(x => Str(x.toString)), TStream(TString), true),
            I32(4)
          )) { inner => ToArray(inner) }
      ),
      IndexedSeq(
        IndexedSeq("0", "1", "2", "3"),
        IndexedSeq("4", "5", "6", "7"),
        IndexedSeq("8", "9")))

  }

  @Test def testMakeStream() {
    assertEvalsTo(
      ToArray(
        MakeStream(IndexedSeq(I32(1), NA(TInt32), I32(2)), TStream(TInt32))
      ),
      IndexedSeq(1, null, 2)
    )

    assertEvalsTo(
      ToArray(
        MakeStream(IndexedSeq(Literal(TArray(TInt32), IndexedSeq(1)), NA(TArray(TInt32))), TStream(TArray(TInt32)))
      ),
      IndexedSeq(IndexedSeq(1), null)
    )
  }

  @Test def testMultiplicity() {
    val target = Ref("target", TStream(TInt32))
    val i = Ref("i", TInt32)
    for ((ir, v) <- Seq(
      StreamRange(0, 10, 1) -> 0,
      target -> 1,
      Let("x", True(), target) -> 1,
      StreamMap(target, "i", i) -> 1,
      StreamMap(StreamMap(target, "i", i), "i", i * i) -> 1,
      StreamFilter(target, "i", StreamFold(StreamRange(0, i, 1), 0, "a", "i", i)) -> 1,
      StreamFilter(StreamRange(0, 5, 1), "i", StreamFold(target, 0, "a", "i", i)) -> 2,
      StreamFlatMap(target, "i", StreamRange(0, i, 1)) -> 1,
      StreamFlatMap(StreamRange(0, 5, 1), "i", target) -> 2,
      StreamScan(StreamMap(target, "i", i), 0, "a", "i", i) -> 1,
      StreamScan(StreamScan(target, 0, "a", "i", i), 0, "a", "i", i) -> 1
    )) {
      assert(StreamUtils.multiplicity(ir, "target") == v, Pretty(ir))
    }
  }
}
