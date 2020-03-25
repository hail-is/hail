package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValue, RegionValueBuilder, SafeRow, ScalaToRegionValue}
import is.hail.asm4s._
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
  private def compile1[T: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Value[T]) => Code[R]): T => R = {
    val fb = EmitFunctionBuilder[T, R]("stream_test")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getArg[T](1)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  private def compile2[T: TypeInfo, U: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Code[T], Code[U]) => Code[R]): (T, U) => R = {
    val fb = EmitFunctionBuilder[T, U, R]("F")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getArg[T](1), mb.getArg[U](2)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  private def compile3[T: TypeInfo, U: TypeInfo, V: TypeInfo, R: TypeInfo](f: (EmitMethodBuilder[_], Code[T], Code[U], Code[V]) => Code[R]): (T, U, V) => R = {
    val fb = EmitFunctionBuilder[T, U, V, R]("F")
    val mb = fb.apply_method
    mb.emit(f(mb, mb.getArg[T](1), mb.getArg[U](2), mb.getArg[V](3)))
    val asmFn = fb.result()()
    asmFn.apply
  }

  def range(start: Code[Int], stop: Code[Int], name: String)(implicit ctx: EmitStreamContext): CodeStream.Stream[Code[Int]] =
    CodeStream.map(CodeStream.range(ctx.mb, start, 1, stop - start))(
      a => a,
      setup0 = Some(Code._println(const(s"$name setup0"))),
      setup = Some(Code._println(const(s"$name setup"))),
      close0 = Some(Code._println(const(s"$name close0"))),
      close = Some(Code._println(const(s"$name close"))))

  class CheckedStream[T](_stream: CodeStream.Stream[T], name: String, mb: EmitMethodBuilder[_]) {
    val outerBit = mb.newLocal[Boolean]()
    val innerBit = mb.newLocal[Boolean]()
    val innerCount = mb.newLocal[Int]()

    def init: Code[Unit] = Code(outerBit := false, innerBit := false, innerCount := 0)

    val stream: CodeStream.Stream[T] = _stream.mapCPS(
      (ctx, a, k) => (outerBit & innerBit).mux(
        k(a),
        Code._fatal[Unit](s"$name: pulled from when not setup")),
      setup0 = Some((!outerBit & !innerBit).mux(
        Code(outerBit := true,
             Code._println(const(s"$name setup0"))),
        Code._fatal[Unit](s"$name: setup0 run out of order"))),
      setup = Some((outerBit & !innerBit).mux(
        Code(innerBit := true,
             innerCount := innerCount.load + 1,
             Code._println(const(s"$name setup"))),
        Code._fatal[Unit](s"$name: setup run out of order"))),
      close0 = Some((outerBit & !innerBit).mux(
        Code(outerBit := false,
             Code._println(const(s"$name close0"))),
        Code._fatal[Unit](s"$name: close0 run out of order"))),
      close = Some((outerBit & innerBit).mux(
        Code(innerBit := false,
             Code._println(const(s"$name close"))),
        Code._fatal[Unit](s"$name: close run out of order"))))

    def assertClosed(expectedRuns: Code[Int]): Code[Unit] =
      Code.memoize(innerCount, "inner_count", expectedRuns, "expected_runs") { (innerCount, expectedRuns) =>
        (outerBit | innerBit).mux(
          Code._fatal[Unit](s"$name: not closed"),
          innerCount.cne(expectedRuns).mux(
            Code._fatal[Unit](const(s"$name: expected ").concat(expectedRuns.toS).concat(" runs, found ").concat(innerCount.toS)),
            Code._empty))
      }

    def assertClosed: Code[Unit] =
      (outerBit | innerBit).mux(
        Code._fatal[Unit](s"$name: not closed"),
        Code._empty)
  }

  def checkedRange(start: Code[Int], stop: Code[Int], name: String, mb: EmitMethodBuilder[_]): CheckedStream[Code[Int]] = {
    val tstart = mb.newLocal[Int]()
    val len = mb.newLocal[Int]()
    val s = CodeStream.range(mb, tstart, 1, len)
      .map(
        f = x => x,
        setup0 = Some(Code(tstart := 0, len := 0)),
        setup = Some(Code(tstart := start, len := stop - tstart)))
    new CheckedStream(s, name, mb)
  }

  @Test def testES2Range() {
    val f = compile1[Int, Unit] { (mb, n) =>
      val r = checkedRange(0, n, "range", mb)

      Code(
        r.init,
        r.stream.forEach(mb)(i => Code._println(i.toS)),
        r.assertClosed(1))
    }
    for (i <- 0 to 2) { f(i) }
  }

  @Test def testES2Zip() {
    val f = compile2[Int, Int, Unit] { (mb, m, n) =>
      val l = checkedRange(0, m, "left", mb)
      val r = checkedRange(0, n, "right", mb)
      val z = CodeStream.zip(l.stream, r.stream)

      Code(
        l.init, r.init,
        z.forEach(mb)(x => Code._println(const("(").concat(x._1.toS).concat(", ").concat(x._2.toS).concat(")"))),
        l.assertClosed(1), r.assertClosed(1))
    }
    for {
      i <- 0 to 2
      j <- 0 to 2
    } {
      f(i, j)
    }
  }

  @Test def testES2FlatMap() {
    val f = compile1[Int, Unit] { (mb, n) =>
      val outer = checkedRange(1, n, "outer", mb)
      var inner: CheckedStream[Code[Int]] = null
      def f(i: Code[Int]) = {
        inner = checkedRange(0, i, "inner", mb)
        inner.stream
      }
      val run = outer.stream.flatMap(f).forEach(mb)(i => Code._println(i.toS))

      Code(
        outer.init, inner.init,
        run,
        outer.assertClosed(1),
        inner.assertClosed(n - 1))
    }
    for (n <- 1 to 5) { f(n) }
  }

  @Test def testES2ZipNested() {
    val f = compile2[Int, Int, Unit] { (mb, m, n) =>
      val l = checkedRange(1, m, "left", mb)

      val rOuter = checkedRange(1, n, "right outer", mb)
      var rInner: CheckedStream[Code[Int]] = null

      def f(i: Code[Int]) = {
        rInner = checkedRange(0, i, "right inner", mb)
        rInner.stream
      }
      val run = CodeStream.zip(l.stream, rOuter.stream.flatMap(f))
                          .forEach(mb)(x => Code._println(const("(").concat(x._1.toS).concat(", ").concat(x._2.toS).concat(")")))

      Code(
        l.init, rOuter.init, rInner.init,
        run,
        l.assertClosed(1),
        rOuter.assertClosed(1),
        rInner.assertClosed)
    }
    f(1, 1)
    f(1, 2)
    f(2, 1)
    f(2, 2)
    f(2, 3)
    f(10, 3)
  }

  @Test def testES2MultiZip() {
    import scala.collection.IndexedSeq
    val f = compile3[Int, Int, Int, Unit] { (mb, n1, n2, n3) =>
      val s1 = checkedRange(0, n1, "s1", mb)
      val s2 = checkedRange(0, n2, "s2", mb)
      val s3 = checkedRange(0, n3, "s3", mb)
      val z = CodeStream.multiZip(IndexedSeq(s1.stream, s2.stream, s3.stream)).asInstanceOf[CodeStream.Stream[IndexedSeq[Code[Int]]]]

      Code(
        s1.init, s2.init, s3.init,
        z.forEach(mb)(x => Code._println(const("(").concat(x(0).toS).concat(", ").concat(x(1).toS).concat(", ").concat(x(2).toS).concat(")"))),
        s1.assertClosed(1),
        s2.assertClosed(1),
        s3.assertClosed(1))
    }
    for {
      n1 <- 0 to 2
      n2 <- 0 to 2
      n3 <- 0 to 2
    } {
      f(n1, n2, n3)
    }
  }

  private def compileStream[F >: Null : TypeInfo, T](
    streamIR: IR,
    inputTypes: Seq[PType]
  )(call: (F, Region, T) => Long): T => IndexedSeq[Any] = {
    val argTypeInfos = new ArrayBuilder[MaybeGenericTypeInfo[_]]
    argTypeInfos += GenericTypeInfo[Region]()
    inputTypes.foreach { t =>
      argTypeInfos ++= Seq(GenericTypeInfo()(typeToTypeInfo(t)), GenericTypeInfo[Boolean]())
    }
    val fb = EmitFunctionBuilder[F]("F", argTypeInfos.result(), GenericTypeInfo[Long])
    val mb = fb.apply_method
    val ir = streamIR.deepCopy()
    InferPType(ir, Env.empty)
    val eltType = ir.pType.asInstanceOf[PStream].elementType
    val stream = ExecuteContext.scoped { ctx =>
      val s = ir match {
        case ToArray(s) => s
        case s => s
      }
      TypeCheck(s)
      EmitStream(new Emit(ctx, fb.ecb), mb, s, Env.empty, EmitRegion.default(mb), None)
    }
    mb.emit {
      val arrayt = EmitStream.toArray(mb, PArray(eltType), stream)
      Code(arrayt.setup, arrayt.m.mux(0L, arrayt.v))
    }
    val f = fb.resultWithIndex()
    (arg: T) => Region.scoped { r =>
      val off = call(f(0, r), r, arg)
      if (off == 0L)
        null
      else
        SafeRow.read(PArray(eltType), off).asInstanceOf[IndexedSeq[Any]]
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
    val ir = streamIR.deepCopy()
    InferPType(ir, Env.empty)
    val optStream = ExecuteContext.scoped { ctx =>
      TypeCheck(ir)
      EmitStream(new Emit(ctx, fb.ecb), mb, ir, Env.empty, EmitRegion.default(mb), None)
    }
    val L = CodeLabel()
    val len = mb.newLocal[Int]()
    implicit val ctx = EmitStreamContext(mb)
    fb.emit(
      Code(
        optStream(
          Code(len := 0, L.goto), { stream =>
            stream.length.map[Code[Ctrl]] { case (s, l) => Code(s, len := l, L.goto) }.getOrElse[Code[Ctrl]](
              Code(len := -1, L.goto))
          }),
        L,
        len))
    val f = fb.resultWithIndex()
    Region.scoped { r =>
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
    val tripleType = PStruct(false, "start" -> PInt32(), "stop" -> PInt32(), "step" -> PInt32())
    val range = compileStream(
      StreamRange(GetField(In(0, tripleType), "start"), GetField(In(0, tripleType), "stop"), GetField(In(0, tripleType), "step")),
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
      val expectedLen = Some(if(v == null) 0 else v.length)
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

  @Test def testEmitLeftJoinDistinct() {
    val tupTyp = TTuple(TInt32, TString)
    def cmp = ApplyComparisonOp(
      Compare(TInt32),
      GetTupleElement(Ref("l", tupTyp), 0),
      GetTupleElement(Ref("r", tupTyp), 0))

    def leftjoin(lstream: IR, rstream: IR): IR =
      StreamLeftJoinDistinct(lstream, rstream,
        "l", "r", cmp,
        MakeTuple.ordered(Seq(
          GetTupleElement(Ref("l", tupTyp), 1),
          GetTupleElement(Ref("r", tupTyp), 1))))

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
    val intsPType = PStream(PInt32())

    val f1 = compileStreamWithIter(
      StreamScan(In(0, intsPType),
        zero = 0,
        "a", "x", Ref("a", TInt32) + Ref("x", TInt32) * Ref("x", TInt32)
      ), intsPType)
    assert(f1((1 to 4).iterator) == IndexedSeq(0, 1, 1+4, 1+4+9, 1+4+9+16))
    assert(f1(Iterator.empty) == IndexedSeq(0))
    assert(f1(null) == null)

    val f2 = compileStreamWithIter(
      StreamFlatMap(
        In(0, intsPType),
        "n", StreamRange(0, Ref("n", TInt32), 1)
      ), intsPType)
    assert(f2(Seq(1, 5, 2, 9).iterator) == IndexedSeq(1, 5, 2, 9).flatMap(0 until _))
    assert(f2(null) == null)
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

    val (pt, f) = Compile[Long, Long](ctx, "in", t, ir)

    Region.smallScoped { r =>
      val rvb = new RegionValueBuilder(r)
      rvb.start(t)
      rvb.addAnnotation(t.virtualType, Row(null, IndexedSeq(1d, 2d), IndexedSeq(3d, 4d)))
      val input = rvb.end()

      assert(SafeRow.read(pt, f(0, r)(r, input, false)) == Row(null))
    }
  }
}
