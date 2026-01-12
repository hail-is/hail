package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.{Region, SafeRow, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.collection.FastSeq
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.agg.{CollectStateSig, PhysicalAggSig, TypedStateSig}
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.ir.streams.{EmitStream, StreamUtils}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.{
  PTypeReferenceSingleCodeType, SingleCodeSCode, StreamSingleCodeType,
}
import is.hail.types.physical.stypes.interfaces.{NoBoxLongIterator, SStreamValue}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.Call2

import org.apache.spark.sql.Row
import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.testng.annotations.Test

class EmitStreamSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

  def log(str: Code[String], enabled: Boolean = false): Code[Unit] =
    if (enabled) Code._println(str) else Code._empty

  private def compileStream[F: TypeInfo, T](
    streamIR: IR,
    inputTypes: IndexedSeq[EmitParamType],
  )(
    call: (F, Region, T) => Long
  ): T => IndexedSeq[Any] = {
    val fb = EmitFunctionBuilder[F](
      ctx,
      "F",
      (classInfo[Region]: ParamType) +: inputTypes.map(pt => pt: ParamType),
      LongInfo,
    )
    val mb = fb.apply_method
    val ir = streamIR.deepCopy()

    val emitContext = EmitContext.analyze(ctx, ir)

    var arrayType: PType = null
    mb.emit(EmitCodeBuilder.scopedCode(mb) { cb =>
      val region = mb.getCodeParam[Region](1)
      val s = ir match {
        case ToArray(s) => s
        case s => s
      }
      TypeCheck(ctx, s)
      EmitStream.produce(
        new Emit(emitContext, fb.ecb),
        s,
        cb,
        cb.emb,
        region,
        EmitEnv(Env.empty, inputTypes.indices.map(i => mb.storeEmitParamAsField(cb, i + 2))),
        None,
      )
        .consumeCode[Long](
          cb,
          0L,
          { s =>
            val arr = StreamUtils.toArray(cb, s.asStream.getProducer(mb), region)
            val scp = SingleCodeSCode.fromSCode(cb, arr, region, false)
            arrayType = scp.typ.asInstanceOf[PTypeReferenceSingleCodeType].pt

            coerce[Long](scp.code)
          },
        )
    })
    val f = fb.resultWithIndex()
    (arg: T) =>
      pool.scopedRegion { r =>
        val off = call(f(theHailClassLoader, ctx.fs, ctx.taskContext, r), r, arg)
        if (off == 0L)
          null
        else
          SafeRow.read(arrayType, off).asInstanceOf[IndexedSeq[Any]]
      }
  }

  private def compileStream(ir: IR, inputType: PType): Any => IndexedSeq[Any] = {
    type F = AsmFunction3RegionLongBooleanLong
    compileStream[F, Any](
      ir,
      FastSeq(SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(inputType))),
    ) { (f: F, r: Region, arg: Any) =>
      if (arg == null)
        f(r, 0L, true)
      else
        f(r, ScalaToRegionValue(ctx.stateManager, r, inputType, arg), false)
    }
  }

  private def compileStreamWithIter(
    ir: IR,
    requiresMemoryManagementPerElement: Boolean,
    elementType: PType,
  ): Iterator[Any] => IndexedSeq[Any] = {
    trait F {
      def apply(o: Region, a: NoBoxLongIterator): Long
    }
    compileStream[F, Iterator[Any]](
      ir,
      IndexedSeq(SingleCodeEmitParamType(
        true,
        StreamSingleCodeType(requiresMemoryManagementPerElement, elementType, true),
      )),
    ) { (f: F, r: Region, it: Iterator[Any]) =>
      val rvi = new NoBoxLongIterator {
        var _eltRegion: Region = _
        var eos: Boolean = _

        override def init(outerRegion: Region, eltRegion: Region): Unit = _eltRegion = eltRegion

        override def next(): Long =
          if (eos || !it.hasNext) {
            eos = true
            0L
          } else
            ScalaToRegionValue(ctx.stateManager, _eltRegion, elementType, it.next())
        override def close(): Unit = ()
      }
      assert(it != null, "null iterators not supported")
      f(r, rvi)
    }
  }

  private def evalStream(ir: IR): IndexedSeq[Any] =
    compileStream[AsmFunction1RegionLong, Unit](ir, FastSeq())((f, r, _) => f(r))
      .apply(())

  private def evalStreamLen(streamIR: IR): Option[Int] = {
    val fb = EmitFunctionBuilder[Region, Int](ctx, "eval_stream_len")
    val mb = fb.apply_method
    val region = mb.getCodeParam[Region](1)
    val ir = streamIR.deepCopy()
    val emitContext = EmitContext.analyze(ctx, ir)

    fb.emitWithBuilder { cb =>
      TypeCheck(ctx, ir)
      val len = cb.newLocal[Int]("len", 0)
      val len2 = cb.newLocal[Int]("len2", -1)

      EmitStream.produce(
        new Emit(emitContext, fb.ecb),
        ir,
        cb,
        cb.emb,
        region,
        EmitEnv(Env.empty, FastSeq()),
        None,
      )
        .consume(
          cb,
          {},
          { case stream: SStreamValue =>
            val producer = stream.getProducer(cb.emb)
            producer.memoryManagedConsume(
              region,
              cb,
              cb => producer.length.foreach(computeLen => cb.assign(len2, computeLen(cb))),
            )(cb => cb.assign(len, len + 1))
          },
        )
      cb.if_(
        len2.cne(-1) && (len2.cne(len)),
        cb._fatal(
          s"length mismatch between computed and iteration length: computed=",
          len2.toS,
          ", iter=",
          len.toS,
        ),
      )

      len2
    }
    val f = fb.resultWithIndex()
    pool.scopedRegion { r =>
      val len = f(theHailClassLoader, ctx.fs, ctx.taskContext, r)(r)
      if (len < 0) None else Some(len)
    }
  }

  @Test def testEmitNA(): Unit =
    assert(evalStream(NA(TStream(TInt32))) == null)

  @Test def testEmitMake(): Unit = {
    val typ = TStream(TInt32)
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      MakeStream(IndexedSeq[IR](1, 2, NA(TInt32), 3), typ) -> IndexedSeq(1, 2, null, 3),
      MakeStream(IndexedSeq[IR](), typ) -> IndexedSeq(),
      MakeStream(
        IndexedSeq[IR](MakeTuple.ordered(IndexedSeq(4, 5))),
        TStream(TTuple(TInt32, TInt32)),
      ) ->
        IndexedSeq(Row(4, 5)),
      MakeStream(IndexedSeq[IR](Str("hi"), Str("world")), TStream(TString)) ->
        IndexedSeq("hi", "world"),
    )
    forAll(tests) { case (ir, v) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir).contains(v.length), Pretty(ctx, ir))
    }
  }

  @Test def testEmitRange(): Unit = {
    val tripleType =
      PCanonicalStruct(false, "start" -> PInt32(), "stop" -> PInt32(), "step" -> PInt32())
    val range = compileStream(
      StreamRange(
        GetField(
          In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))),
          "start",
        ),
        GetField(
          In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))),
          "stop",
        ),
        GetField(
          In(0, SingleCodeEmitParamType(false, PTypeReferenceSingleCodeType(tripleType))),
          "step",
        ),
      ),
      tripleType,
    )
    for {
      start <- -2 to 2
      stop <- -2 to 8
      step <- 1 to 3
    }
      assert(
        range(Row(start, stop, step)) == Array.range(start, stop, step).toFastSeq,
        s"($start, $stop, $step)",
      )
    assert(range(Row(null, 10, 1)) == null)
    assert(range(Row(0, null, 1)) == null)
    assert(range(Row(0, 10, null)) == null)
    assert(range(null) == null)
  }

  @Test def testEmitSeqSample(): Unit = {
    val N = 20
    val n = 2

    val seqIr = SeqSample(
      I32(N),
      I32(n),
      RNGStateLiteral(),
      false,
    )

    val compiled = compileStream[AsmFunction1RegionLong, Unit](seqIr, FastSeq()) { (f, r, _) =>
      f(r)
    }

    // Generate many pairs of numbers between 0 and N, every pair should be equally likely
    val results = Array.tabulate(N, N) { case (_, _) => 0 }
    (0 until 1000000).foreach { i =>
      val IndexedSeq = compiled.apply(()).map(_.asInstanceOf[Int])
      assert(IndexedSeq.size == n)
      results(IndexedSeq(0))(IndexedSeq(1)) += 1
      assert(IndexedSeq.toSet.size == n)
      assert(IndexedSeq.forall(e => e >= 0 && e < N))
    }

    forAll(0 until N) { i =>
      forAll(i + 1 until N) { j =>
        val entry = results(i)(j)
        // Expected value of entry is 5263.
        assert(entry > 4880 && entry < 5650)
      }
    }
  }

  @Test def testEmitToStream(): Unit = {
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      ToStream(MakeArray(IndexedSeq[IR](), TArray(TInt32))) -> IndexedSeq(),
      ToStream(MakeArray(IndexedSeq[IR](1, 2, 3, 4), TArray(TInt32))) -> IndexedSeq(1, 2, 3, 4),
      ToStream(NA(TArray(TInt32))) -> null,
    )
    forAll(tests) { case (ir, v) =>
      val expectedLen = Option(v).map(_.length)
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir) == expectedLen, Pretty(ctx, ir))
    }
  }

  @Test def testEmitLet(): Unit = {
    val ir =
      bindIRs(3, 10) { case Seq(start, end) =>
        flatMapIR(StreamRange(start, end, 1)) { i =>
          MakeStream(IndexedSeq(i, end), TStream(TInt32))
        }
      }
    assert(evalStream(ir) == (3 until 10).flatMap(i => IndexedSeq(i, 10)), Pretty(ctx, ir))
    assert(evalStreamLen(ir).isEmpty, Pretty(ctx, ir))
  }

  @Test def testEmitMap(): Unit = {
    def ten = StreamRange(I32(0), I32(10), I32(1))

    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      mapIR(ten)(_ * 2) -> (0 until 10).map(_ * 2),
      mapIR(ten)(_.toL) -> (0 until 10).map(_.toLong),
      mapIR(mapIR(ten)(_ + 1))(y => y * y) -> (0 until 10).map(i => (i + 1) * (i + 1)),
      mapIR(ten)(_ => NA(TInt32)) -> IndexedSeq.tabulate(10)(_ => null),
    )
    forAll(tests) { case (ir, v) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir).contains(v.length), Pretty(ctx, ir))
    }
  }

  @Test def testEmitFilter(): Unit = {
    def ten = StreamRange(I32(0), I32(10), I32(1))

    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      filterIR(ten)(_ cne 5) -> (0 until 10).filter(_ != 5),
      filterIR(mapIR(ten)(x => (x * 2).toL))(_ > 5L) ->
        (3 until 10).map(x => (x * 2).toLong),
      filterIR(mapIR(ten)(x => (x * 2).toL))(_ => NA(TBoolean)) -> IndexedSeq(),
      filterIR(mapIR(ten)(_ => NA(TInt32)))(_ => True()) ->
        IndexedSeq.tabulate(10)(_ => null),
    )
    forAll(tests) { case (ir, v) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir).isEmpty, Pretty(ctx, ir))
    }
  }

  @Test def testEmitFlatMap(): Unit = {

    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      flatMapIR(rangeIR(6))(rangeIR(_)) ->
        (0 until 6).flatMap(0 until _),
      flatMapIR(rangeIR(6))(_ => rangeIR(NA(TInt32))) ->
        IndexedSeq(),
      flatMapIR(rangeIR(NA(TInt32)))(rangeIR(_)) -> null,
      flatMapIR(rangeIR(20))(x => flatMapIR(rangeIR(x))(y => rangeIR(x + y))) ->
        (0 until 20).flatMap(x => (0 until x).flatMap(y => 0 until (x + y))),
      flatMapIR(filterIR(rangeIR(5))(_ cne 3)) { y =>
        MakeStream(IndexedSeq(y, y), TStream(TInt32))
      } -> IndexedSeq(0, 0, 1, 1, 2, 2, 4, 4),
      flatMapIR(rangeIR(4)) { x =>
        ToStream(MakeArray(IndexedSeq[IR](x, x), TArray(TInt32)))
      } -> IndexedSeq(0, 0, 1, 1, 2, 2, 3, 3),
    )

    forAll(tests) { case (ir, v) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      if (v != null)
        assert(evalStreamLen(ir).isEmpty, Pretty(ctx, ir))

    }
  }

  @Test def testStreamBufferedAggregator(): Unit = {
    val resultArrayToCompare = (0 until 12).map(i => Row(Row(i, i + 1), 1))
    val streamType = TStream(TStruct("a" -> TInt64, "b" -> TInt64))
    val numSeq = (0L until 12).map(i => IndexedSeq(I64(i), I64(i + 1)))
    val numTupleSeq = numSeq.map(_ => IndexedSeq("a", "b")).zip(numSeq)
    val countStructSeq = numTupleSeq.map { case (s, i) => s.zip(i) }.map(is => MakeStruct(is))
    val countStructStream = MakeStream(countStructSeq, streamType, false)
    val countAggSig = PhysicalAggSig(
      Count(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)),
    )
    val initOps = InitOp(0, FastSeq(), countAggSig)
    val seqOps = SeqOp(0, FastSeq(), countAggSig)
    val foo = Ref(freshName(), streamType.elementType)
    val newKey = MakeStruct(IndexedSeq("count" -> SelectFields(
      foo,
      IndexedSeq("a", "b"),
    )))
    val streamBuffAggCount = StreamBufferedAggregate(
      countStructStream,
      initOps,
      newKey,
      seqOps,
      foo.name,
      IndexedSeq(countAggSig),
      8,
    )
    val result = mapIR(streamBuffAggCount) { elem =>
      MakeStruct(IndexedSeq(
        "key" -> GetField(elem, "count"),
        "aggResult" ->
          RunAgg(
            InitFromSerializedValue(
              0,
              GetTupleElement(GetField(elem, "agg"), 0),
              countAggSig.state,
            ),
            ResultOp(0, countAggSig),
            IndexedSeq(countAggSig.state),
          ),
      ))
    }
    assert(evalStream(result).equals(resultArrayToCompare))

  }

  @Test def testStreamBufferedAggregatorCombine(): Unit = {
    val resultArrayToCompare = IndexedSeq(Row(Row(1), 2))
    val streamType = TStream(TStruct("a" -> TInt64))
    val elemOne = MakeStruct(IndexedSeq(("a", I64(1))))
    val elemTwo = MakeStruct(IndexedSeq(("a", I64(1))))
    val countStructStream = MakeStream(IndexedSeq(elemOne, elemTwo), streamType)
    val countAggSig = PhysicalAggSig(
      Count(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)),
    )
    val initOps = InitOp(0, FastSeq(), countAggSig)
    val seqOps = SeqOp(0, FastSeq(), countAggSig)
    val foo = Ref(freshName(), streamType.elementType)
    val newKey = MakeStruct(IndexedSeq("count" -> SelectFields(
      foo,
      IndexedSeq("a"),
    )))
    val streamBuffAggCount = StreamBufferedAggregate(
      countStructStream,
      initOps,
      newKey,
      seqOps,
      foo.name,
      IndexedSeq(countAggSig),
      8,
    )
    val result = mapIR(streamBuffAggCount) { elem =>
      MakeStruct(IndexedSeq(
        "key" -> GetField(elem, "count"),
        "aggResult" ->
          RunAgg(
            InitFromSerializedValue(
              0,
              GetTupleElement(GetField(elem, "agg"), 0),
              countAggSig.state,
            ),
            ResultOp(0, countAggSig),
            IndexedSeq(countAggSig.state),
          ),
      ))
    }
    assert(evalStream(result) == resultArrayToCompare)
  }

  @Test def testStreamBufferedAggregatorCollectAggregator(): Unit = {
    val resultArrayToCompare =
      IndexedSeq(Row(Row(1), IndexedSeq(1, 3)), Row(Row(2), IndexedSeq(2, 4)))
    val streamType = TStream(TStruct("a" -> TInt64, "b" -> TInt64))
    val elemOne = MakeStruct(IndexedSeq(("a", I64(1)), ("b", I64(1))))
    val elemTwo = MakeStruct(IndexedSeq(("a", I64(2)), ("b", I64(2))))
    val elemThree = MakeStruct(IndexedSeq(("a", I64(1)), ("b", I64(3))))
    val elemFour = MakeStruct(IndexedSeq(("a", I64(2)), ("b", I64(4))))
    val collectStructStream =
      MakeStream(IndexedSeq(elemOne, elemTwo, elemThree, elemFour), streamType)
    val collectAggSig =
      PhysicalAggSig(Collect(), CollectStateSig(VirtualTypeWithReq(PType.canonical(TInt64))))
    val initOps = InitOp(0, FastSeq(), collectAggSig)
    val foo = Ref(freshName(), streamType.elementType)
    val seqOps = SeqOp(0, FastSeq(GetField(foo, "b")), collectAggSig)
    val newKey = MakeStruct(IndexedSeq("collect" -> SelectFields(
      foo,
      IndexedSeq("a"),
    )))
    val streamBuffAggCollect = StreamBufferedAggregate(
      collectStructStream,
      initOps,
      newKey,
      seqOps,
      foo.name,
      IndexedSeq(collectAggSig),
      8,
    )
    val result = mapIR(streamBuffAggCollect) { elem =>
      MakeStruct(IndexedSeq(
        "key" -> GetField(elem, "collect"),
        "aggResult" ->
          RunAgg(
            InitFromSerializedValue(
              0,
              GetTupleElement(GetField(elem, "agg"), 0),
              collectAggSig.state,
            ),
            ResultOp(0, collectAggSig),
            IndexedSeq(collectAggSig.state),
          ),
      ))
    }
    assert(evalStream(result) == resultArrayToCompare)
  }

  @Test def testStreamBufferedAggregatorMultipleAggregators(): Unit = {
    val resultArrayToCompare = IndexedSeq(
      Row(Row(1), Row(3, IndexedSeq(1L, 3L, 2L))),
      Row(Row(2), Row(2, IndexedSeq(2L, 4L))),
      Row(Row(3), Row(3, IndexedSeq(1L, 2L, 3L))),
      Row(Row(4), Row(1, IndexedSeq(4L))),
      Row(Row(5), Row(1, IndexedSeq(1L))),
      Row(Row(6), Row(1, IndexedSeq(3L))),
      Row(Row(7), Row(1, IndexedSeq(4L))),
      Row(Row(8), Row(1, IndexedSeq(1L))),
      Row(Row(8), Row(1, IndexedSeq(2L))),
      Row(Row(9), Row(1, IndexedSeq(3L))),
      Row(Row(10), Row(2, IndexedSeq(4L, 4L))),
    )
    val streamType = TStream(TStruct("a" -> TInt64, "b" -> TInt64))
    val elemOne = MakeStruct(IndexedSeq(("a", I64(1)), ("b", I64(1))))
    val elemTwo = MakeStruct(IndexedSeq(("a", I64(2)), ("b", I64(2))))
    val elemThree = MakeStruct(IndexedSeq(("a", I64(1)), ("b", I64(3))))
    val elemFour = MakeStruct(IndexedSeq(("a", I64(2)), ("b", I64(4))))
    val elemFive = MakeStruct(IndexedSeq(("a", I64(3)), ("b", I64(1))))
    val elemSix = MakeStruct(IndexedSeq(("a", I64(3)), ("b", I64(2))))
    val elemSeven = MakeStruct(IndexedSeq(("a", I64(3)), ("b", I64(3))))
    val elemEight = MakeStruct(IndexedSeq(("a", I64(4)), ("b", I64(4))))
    val elemNine = MakeStruct(IndexedSeq(("a", I64(5)), ("b", I64(1))))
    val elemTen = MakeStruct(IndexedSeq(("a", I64(1)), ("b", I64(2))))
    val elemEleven = MakeStruct(IndexedSeq(("a", I64(6)), ("b", I64(3))))
    val elemTwelve = MakeStruct(IndexedSeq(("a", I64(7)), ("b", I64(4))))
    val elemThirteen = MakeStruct(IndexedSeq(("a", I64(8)), ("b", I64(1))))
    val elemFourteen = MakeStruct(IndexedSeq(("a", I64(8)), ("b", I64(2))))
    val elemFifteen = MakeStruct(IndexedSeq(("a", I64(9)), ("b", I64(3))))
    val elemSixteen = MakeStruct(IndexedSeq(("a", I64(10)), ("b", I64(4))))
    val elemSeventeen = MakeStruct(IndexedSeq(("a", I64(10)), ("b", I64(4))))
    val collectStructStream = MakeStream(
      IndexedSeq(elemOne, elemTwo, elemThree, elemFour, elemFive, elemSix, elemSeven,
        elemEight, elemNine, elemTen, elemEleven, elemTwelve, elemThirteen,
        elemFourteen, elemFifteen, elemSixteen, elemSeventeen),
      streamType,
    )
    val collectAggSig =
      PhysicalAggSig(Collect(), CollectStateSig(VirtualTypeWithReq(PType.canonical(TInt64))))
    val countAggSig = PhysicalAggSig(
      Count(),
      TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true)),
    )
    val initOps = Begin(IndexedSeq(
      InitOp(0, FastSeq(), countAggSig),
      InitOp(1, FastSeq(), collectAggSig),
    ))
    val foo = Ref(freshName(), streamType.elementType)
    val seqOps = Begin(IndexedSeq(
      SeqOp(0, FastSeq(), countAggSig),
      SeqOp(1, FastSeq(GetField(foo, "b")), collectAggSig),
    ))
    val newKey = MakeStruct(IndexedSeq("collect" -> SelectFields(
      foo,
      IndexedSeq("a"),
    )))
    val streamBuffAggCollect = StreamBufferedAggregate(
      collectStructStream,
      initOps,
      newKey,
      seqOps,
      foo.name,
      IndexedSeq(countAggSig, collectAggSig),
      8,
    )
    val result = mapIR(streamBuffAggCollect) { elem =>
      MakeStruct(IndexedSeq(
        "key" -> GetField(elem, "collect"),
        "aggResult" ->
          RunAgg(
            Begin(IndexedSeq(
              InitFromSerializedValue(
                0,
                GetTupleElement(GetField(elem, "agg"), 0),
                countAggSig.state,
              ),
              InitFromSerializedValue(
                1,
                GetTupleElement(GetField(elem, "agg"), 1),
                collectAggSig.state,
              ),
            )),
            MakeTuple.ordered(IndexedSeq(ResultOp(0, countAggSig), ResultOp(1, collectAggSig))),
            IndexedSeq(countAggSig.state, collectAggSig.state),
          ),
      ))
    }
    assert(evalStream(result) == resultArrayToCompare)
  }

  @Test def testEmitJoinRightDistinct(): Unit = {
    val eltType = TStruct("k" -> TInt32, "v" -> TString)

    def join(lstream: IR, rstream: IR, joinType: String): IR =
      joinRightDistinctIR(lstream, rstream, FastSeq("k"), FastSeq("k"), joinType) { (l, r) =>
        MakeTuple.ordered(IndexedSeq(GetField(l, "v"), GetField(r, "v")))
      }

    def leftjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "left")

    def outerjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "outer")

    def pairs(xs: IndexedSeq[(Int, String)]): IR =
      MakeStream(
        xs.map { case (a, b) => MakeStruct(IndexedSeq("k" -> I32(a), "v" -> Str(b))) },
        TStream(eltType),
      )

    val tests: Array[(IR, IR, IndexedSeq[Any], IndexedSeq[Any])] = Array(
      (pairs(IndexedSeq()), pairs(IndexedSeq()), IndexedSeq(), IndexedSeq()),
      (
        pairs(IndexedSeq(3 -> "A")),
        pairs(IndexedSeq()),
        IndexedSeq(Row("A", null)),
        IndexedSeq(Row("A", null)),
      ),
      (pairs(IndexedSeq()), pairs(IndexedSeq(3 -> "B")), IndexedSeq(), IndexedSeq(Row(null, "B"))),
      (
        pairs(IndexedSeq(0 -> "A")),
        pairs(IndexedSeq(0 -> "B")),
        IndexedSeq(Row("A", "B")),
        IndexedSeq(Row("A", "B")),
      ),
      (
        pairs(IndexedSeq(0 -> "A", 2 -> "B", 3 -> "C")),
        pairs(IndexedSeq(0 -> "a", 1 -> ".", 2 -> "b", 4 -> "..")),
        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", null)),
        IndexedSeq(Row("A", "a"), Row(null, "."), Row("B", "b"), Row("C", null), Row(null, "..")),
      ),
      (
        pairs(IndexedSeq(0 -> "A", 1 -> "B1", 1 -> "B2")),
        pairs(IndexedSeq(0 -> "a", 1 -> "b", 2 -> "c")),
        IndexedSeq(Row("A", "a"), Row("B1", "b"), Row("B2", "b")),
        IndexedSeq(Row("A", "a"), Row("B1", "b"), Row("B2", "b"), Row(null, "c")),
      ),
    )
    forAll(tests) { case (lstream, rstream, expectedLeft, expectedOuter) =>
      val l = leftjoin(lstream, rstream)
      val o = outerjoin(lstream, rstream)
      assert(evalStream(l) == expectedLeft, Pretty(ctx, l))
      assert(evalStream(o) == expectedOuter, Pretty(ctx, o))
      assert(evalStreamLen(l).contains(expectedLeft.length), Pretty(ctx, l))
      assert(evalStreamLen(o).isEmpty, Pretty(ctx, o))
    }
  }

  @Test def testEmitJoinRightDistinctInterval(): Unit = {
    val lEltType = TStruct("k" -> TInt32, "v" -> TString)
    val rEltType = TStruct("k" -> TInterval(TInt32), "v" -> TString)

    def join(lstream: IR, rstream: IR, joinType: String): IR =
      joinRightDistinctIR(lstream, rstream, FastSeq("k"), FastSeq("k"), joinType) { (l, r) =>
        MakeTuple.ordered(IndexedSeq(GetField(l, "v"), GetField(r, "v")))
      }

    def leftjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "left")

    def innerjoin(lstream: IR, rstream: IR): IR = join(lstream, rstream, "inner")

    def lElts(xs: (Int, String)*): IR =
      MakeStream(
        xs.toArray.map { case (a, b) => MakeStruct(IndexedSeq("k" -> I32(a), "v" -> Str(b))) },
        TStream(lEltType),
      )

    def rElts(xs: ((Char, Any, Any, Char), String)*): IR =
      MakeStream(
        xs.toArray.map {
          case ((is, s, e, ie), v) =>
            val start = if (s == null) NA(TInt32) else I32(s.asInstanceOf[Int])
            val end = if (e == null) NA(TInt32) else I32(e.asInstanceOf[Int])
            val includesStart = is == '['
            val includesEnd = ie == ']'
            val interval = ApplySpecial(
              "Interval",
              FastSeq(),
              FastSeq(start, end, includesStart, includesEnd),
              TInterval(TInt32),
              0,
            )
            MakeStruct(IndexedSeq("k" -> interval, "v" -> Str(v)))
        },
        TStream(rEltType),
      )

    val tests: Array[(IR, IR, IndexedSeq[Any], IndexedSeq[Any])] = Array(
      (lElts(), rElts(), IndexedSeq(), IndexedSeq()),
      (lElts(3 -> "A"), rElts(), IndexedSeq(Row("A", null)), IndexedSeq()),
      (lElts(), rElts(('[', 1, 2, ']') -> "B"), IndexedSeq(), IndexedSeq()),
      (
        lElts(0 -> "A"),
        rElts(('[', 0, 1, ')') -> "B"),
        IndexedSeq(Row("A", "B")),
        IndexedSeq(Row("A", "B")),
      ),
      (lElts(0 -> "A"), rElts(('(', 0, 1, ')') -> "B"), IndexedSeq(Row("A", null)), IndexedSeq()),
      (
        lElts(0 -> "A", 2 -> "B", 3 -> "C", 4 -> "D"),
        rElts(
          ('[', 0, 2, ')') -> "a",
          ('(', 0, 1, ']') -> ".",
          ('[', 1, 4, ')') -> "b",
          ('[', 2, 4, ')') -> "..",
        ),
        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", "b"), Row("D", null)),
        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", "b")),
      ),
      (
        lElts(1 -> "A", 2 -> "B", 3 -> "C", 4 -> "D"),
        rElts(
          ('[', 0, null, ')') -> ".",
          ('(', 0, 1, ']') -> "a",
          ('[', 1, 4, ')') -> "b",
          ('[', 2, 4, ')') -> "..",
        ),
        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", "b"), Row("D", null)),
        IndexedSeq(Row("A", "a"), Row("B", "b"), Row("C", "b")),
      ),
    )

    forAll(tests) { case (lstream, rstream, expectedLeft, expectedInner) =>
      val l = leftjoin(lstream, rstream)
      val i = innerjoin(lstream, rstream)
      assert(evalStream(l) == expectedLeft, Pretty(ctx, l))
      assert(evalStream(i) == expectedInner, Pretty(ctx, i))
      assert(evalStreamLen(l).contains(expectedLeft.length), Pretty(ctx, l))
      assert(evalStreamLen(i).isEmpty, Pretty(ctx, i))
    }
  }

  @Test def testStreamJoinOuterWithKeyRepeats(): Unit = {
    val lEltType = TStruct("k" -> TInt32, "idx_left" -> TInt32)
    val lRows = FastSeq(
      Row(1, 1),
      Row(1, 2),
      Row(1, 3),
      Row(3, 4),
    )

    val a = ToStream(Literal(TArray(lEltType), lRows))

    val rEltType = TStruct("k" -> TInt32, "idx_right" -> TInt32)
    val rRows = FastSeq(
      Row(1, 1),
      Row(2, 2),
      Row(4, 3),
    )
    val b = ToStream(Literal(TArray(rEltType), rRows))

    val ir = joinRightDistinctIR(a, b, FastSeq("k"), FastSeq("k"), "outer") { (l, r) =>
      MakeStruct(FastSeq("left" -> l, "right" -> r))
    }

    val compiled = evalStream(ir)
    val expected = FastSeq(
      Row(Row(1, 1), Row(1, 1)),
      Row(Row(1, 2), Row(1, 1)),
      Row(Row(1, 3), Row(1, 1)),
      Row(null, Row(2, 2)),
      Row(Row(3, 4), null),
      Row(null, Row(4, 3)),
    )
    assert(compiled == expected)
  }

  @Test def testEmitScan(): Unit = {
    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      streamScanIR(MakeStream(IndexedSeq(), TStream(TInt32)), 9)(_ + _) -> IndexedSeq(9),
      streamScanIR(mapIR(rangeIR(4))(x => x * x), 1)(_ + _) ->
        IndexedSeq(1, 1 /*1+0*0*/, 2 /*1+1*1*/, 6 /*2+2*2*/, 15 /*6+3*3*/ ),
    )
    forAll(tests) { case (ir, v) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir).contains(v.length), Pretty(ctx, ir))
    }
  }

  @Test def testEmitAggScan(): Unit = {
    def assertAggScan(ir: IR, inType: Type, tests: (Any, Any)*): Unit =
      unoptimized { ctx =>
        val aggregate = compileStream(
          LoweringPipeline.compileLowerer(ctx, ir).asInstanceOf[IR],
          PType.canonical(inType),
        )
        forAll(tests) { case (inp, expected) =>
          assert(aggregate(inp) == expected, Pretty(ctx, ir))
        }
      }

    val pairType = TStruct("x" -> TCall, "y" -> TInt32)
    val intsType = TArray(TInt32)

    assertAggScan(
      streamAggScanIR(ToStream(In(0, TArray(pairType)))) { foo =>
        GetField(foo, "y") +
          GetField(
            ApplyScanOp(CallStats(), I32(2))(GetField(foo, "x")),
            "AN",
          )
      },
      TArray(pairType),
      FastSeq(
        Row(null, 1),
        Row(Call2(0, 0), 2),
        Row(Call2(0, 1), 3),
        Row(Call2(1, 1), 4),
        null,
        Row(null, 5),
      ) -> FastSeq(1 + 0, 2 + 0, 3 + 2, 4 + 4, null, 5 + 6),
    )

    assertAggScan(
      streamAggScanIR(
        streamAggScanIR(ToStream(In(0, intsType)))(i => ApplyScanOp(Sum())(i.toL))
      )(x => ApplyScanOp(Max())(x)),
      intsType,
      FastSeq(2, 5, 8, -3, 2, 2, 1, 0, 0) ->
        IndexedSeq(null, 0L, 2L, 7L, 15L, 15L, 15L, 16L, 17L),
    )
  }

  @Test def testEmitFromIterator(): Unit = {
    val intsPType = PInt32(true)

    val f1 = compileStreamWithIter(
      streamScanIR(
        In(0, SingleCodeEmitParamType(true, StreamSingleCodeType(true, PInt32(true), true))),
        zero = 0,
      )((a, x) => a + x * x),
      false,
      intsPType,
    )
    assert(f1((1 to 4).iterator) == IndexedSeq(0, 1, 1 + 4, 1 + 4 + 9, 1 + 4 + 9 + 16))
    assert(f1(Iterator.empty) == IndexedSeq(0))

    val f2 = compileStreamWithIter(
      flatMapIR(
        In(0, SingleCodeEmitParamType(true, StreamSingleCodeType(false, PInt32(true), true)))
      )(n => StreamRange(0, n, 1)),
      false,
      intsPType,
    )
    assert(f2(IndexedSeq(1, 5, 2, 9).iterator) == IndexedSeq(1, 5, 2, 9).flatMap(0 until _))

    val f3 = compileStreamWithIter(
      rangeIR(
        StreamLen(In(
          0,
          SingleCodeEmitParamType(true, StreamSingleCodeType(false, PInt32(true), true)),
        ))
      ),
      false,
      intsPType,
    )
    assert(f3(IndexedSeq(1, 5, 2, 9).iterator) == IndexedSeq(0, 1, 2, 3))
    assert(f3(IndexedSeq().iterator) == IndexedSeq())
  }

  @Test def testEmitIf(): Unit = {
    def xs = MakeStream(IndexedSeq[IR](5, 3, 6), TStream(TInt32))

    def ys = StreamRange(0, 4, 1)

    def na = NA(TStream(TInt32))

    val tests: Array[(IR, IndexedSeq[Any])] = Array(
      If(True(), xs, ys) -> IndexedSeq(5, 3, 6),
      If(False(), xs, ys) -> IndexedSeq(0, 1, 2, 3),
      If(True(), xs, na) -> IndexedSeq(5, 3, 6),
      If(False(), xs, na) -> null,
      If(NA(TBoolean), xs, ys) -> null,
      flatMapIR(
        MakeStream(IndexedSeq(False(), True(), False()), TStream(TBoolean))
      )(x => If(x, xs, ys))
        -> IndexedSeq(0, 1, 2, 3, 5, 3, 6, 0, 1, 2, 3),
    )
    val lens: Array[Option[Int]] = Array(Some(3), Some(4), Some(3), None, None, None)
    forAll(tests zip lens) { case ((ir, v), len) =>
      assert(evalStream(ir) == v, Pretty(ctx, ir))
      assert(evalStreamLen(ir) == len, Pretty(ctx, ir))
    }
  }

  @Test def testZipIfNA(): Unit = {

    val t = PCanonicalStruct(
      true,
      "missingParam" -> PCanonicalArray(PFloat64()),
      "xs" -> PCanonicalArray(PFloat64()),
      "ys" -> PCanonicalArray(PFloat64()),
    )
    val i1 = Ref(freshName(), t.virtualType)
    val ir = MakeTuple.ordered(IndexedSeq(foldIR(
      zip2(
        ToStream(If(
          IsNA(GetField(i1, "missingParam")),
          NA(TArray(TFloat64)),
          GetField(i1, "xs"),
        )),
        ToStream(GetField(i1, "ys")),
        ArrayZipBehavior.AssertSameLength,
      )(_ * _),
      F64(0d),
    )(_ + _)))

    val (Some(PTypeReferenceSingleCodeType(pt)), f) = Compile[AsmFunction2RegionLongLong](
      ctx,
      FastSeq((i1.name, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(t)))),
      FastSeq(classInfo[Region], LongInfo),
      LongInfo,
      ir,
    )

    pool.scopedSmallRegion { r =>
      val input = t.unstagedStoreJavaObject(
        ctx.stateManager,
        Row(null, IndexedSeq(1d, 2d), IndexedSeq(3d, 4d)),
        r,
      )

      assert(
        SafeRow.read(pt, f(theHailClassLoader, ctx.fs, ctx.taskContext, r)(r, input)) == Row(null)
      )
    }
  }

  @Test def testFold(): Unit = {
    val ints = Literal(TArray(TInt32), IndexedSeq(1, 2, 3, 4))
    val strsLit = Literal(TArray(TString), IndexedSeq("one", "two", "three", "four"))
    val strs =
      MakeStream(FastSeq(Str("one"), Str("two"), Str("three"), Str("four")), TStream(TString), true)

    assertEvalsTo(
      foldIR(ToStream(ints, requiresMemoryManagementPerElement = false), I32(-1)) { (acc, elt) =>
        acc + elt
      },
      9,
    )

    assertEvalsTo(
      foldIR(ToStream(strsLit, requiresMemoryManagementPerElement = false), Str("")) { (acc, elt) =>
        invoke("concat", TString, acc, elt)
      },
      "onetwothreefour",
    )

    assertEvalsTo(
      foldIR(strs, Str(""))((acc, elt) => invoke("concat", TString, acc, elt)),
      "onetwothreefour",
    )
  }

  @Test def testGrouped(): Unit = {
    // empty => empty
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 0, 1, false),
            I32(5),
          )
        )(inner => ToArray(inner))
      ),
      IndexedSeq(),
    )

    // general case where stream ends in inner group
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 10, 1, false),
            I32(3),
          )
        )(inner => ToArray(inner))
      ),
      IndexedSeq(
        IndexedSeq(0, 1, 2),
        IndexedSeq(3, 4, 5),
        IndexedSeq(6, 7, 8),
        IndexedSeq(9),
      ),
    )

    // stream ends at end of inner group
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            StreamRange(0, 10, 1, false),
            I32(5),
          )
        )(inner => ToArray(inner))
      ),
      IndexedSeq(
        IndexedSeq(0, 1, 2, 3, 4),
        IndexedSeq(5, 6, 7, 8, 9),
      ),
    )

    // separate regions
    assertEvalsTo(
      ToArray(
        mapIR(
          StreamGrouped(
            MakeStream((0 until 10).map(x => Str(x.toString)), TStream(TString), true),
            I32(4),
          )
        )(inner => ToArray(inner))
      ),
      IndexedSeq(
        IndexedSeq("0", "1", "2", "3"),
        IndexedSeq("4", "5", "6", "7"),
        IndexedSeq("8", "9"),
      ),
    )

  }

  @Test def testMakeStream(): Unit = {
    assertEvalsTo(
      ToArray(
        MakeStream(IndexedSeq(I32(1), NA(TInt32), I32(2)), TStream(TInt32))
      ),
      IndexedSeq(1, null, 2),
    )

    assertEvalsTo(
      ToArray(
        MakeStream(
          IndexedSeq(Literal(TArray(TInt32), IndexedSeq(1)), NA(TArray(TInt32))),
          TStream(TArray(TInt32)),
        )
      ),
      IndexedSeq(IndexedSeq(1), null),
    )
  }

  @Test def testMultiplicity(): Unit = {
    val target = Ref(freshName(), TStream(TInt32))
    val tests = IndexedSeq(
      StreamRange(0, 10, 1) -> 0,
      target -> 1,
      bindIR(True())(_ => target) -> 1,
      mapIR(target)(i => i) -> 1,
      mapIR(mapIR(target)(i => i))(i => i * i) -> 1,
      filterIR(target)(i => foldIR(rangeIR(i), 0)((_, i) => i)) -> 1,
      filterIR(rangeIR(5))(_ => foldIR(target, 0)((_, i) => i)) -> 2,
      flatMapIR(target)(i => rangeIR(i)) -> 1,
      flatMapIR(rangeIR(5))(_ => target) -> 2,
      streamScanIR(mapIR(target)(i => i), 0)((_, i) => i) -> 1,
      streamScanIR(streamScanIR(target, 0)((_, i) => i), 0)((_, i) => i) -> 1,
    )

    forAll(tests) { case (ir, v) =>
      assert(StreamUtils.multiplicity(ir, target.name) == v, Pretty(ctx, ir))
    }
  }

  def assertMemoryDoesNotScaleWithStreamSize(lowSize: Int = 50, highSize: Int = 2500)(f: IR => IR)
    : Unit =
    unoptimized { ctx =>
      val (_, memUsed1) =
        measuringHighestTotalMemoryUsage(ctx => eval(f(lowSize), ctx = ctx))

      val (_, memUsed2) =
        measuringHighestTotalMemoryUsage(ctx => eval(f(highSize), ctx = ctx))

      assert(
        memUsed1 == memUsed2,
        s"memory usage scales with stream size!" +
          s"\n  at size=$lowSize, memory=$memUsed1" +
          s"\n  at size=$highSize, memory=$memUsed2" +
          s"\n  IR: ${Pretty(ctx, f(lowSize))}",
      )
    }

  def sumIR(x: IR): IR = foldIR(x, 0) { case (acc, value) => acc + value }

  def foldLength(x: IR): IR = sumIR(mapIR(x)(_ => I32(1)))

  def rangeStructs(size: IR): IR = mapIR(StreamRange(0, size, 1, true)) { i =>
    makestruct(
      ("idx", i),
      ("foo", invoke("str", TString, i)),
      ("bigArray", ToArray(rangeIR(10000))),
    )
  }

  def filteredRangeStructs(size: IR): IR = mapIR(filterIR(
    StreamRange(0, size, 1, true)
  )(i => i < (size / 2).toI)) { i =>
    makestruct(
      ("idx", i),
      ("foo2", invoke("str", TString, i)),
      ("bigArray2", ToArray(rangeIR(10000))),
    )
  }

  @Test def testMemoryRangeFold(): Unit =
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldIR(
        mapIR(flatMapIR(StreamRange(0, size, 1, true))(x => StreamRange(0, x, 1, true))) { i =>
          invoke("str", TString, i)
        },
        I32(0),
      ) { case (acc, value) => maxIR(acc, invoke("length", TInt32, value)) }
    }

  @Test def testStreamJoinMemory(): Unit = {

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(joinRightDistinctIR(
        rangeStructs(size),
        filteredRangeStructs(size),
        IndexedSeq("idx"),
        IndexedSeq("idx"),
        "inner",
      ) { case (_, _) => I32(1) })
    }
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(joinRightDistinctIR(
        rangeStructs(size),
        filteredRangeStructs(size),
        IndexedSeq("idx"),
        IndexedSeq("idx"),
        "left",
      ) { case (_, _) => I32(1) })
    }
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(joinRightDistinctIR(
        rangeStructs(size),
        filteredRangeStructs(size),
        IndexedSeq("idx"),
        IndexedSeq("idx"),
        "right",
      ) { case (_, _) => I32(1) })
    }
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(joinRightDistinctIR(
        rangeStructs(size),
        filteredRangeStructs(size),
        IndexedSeq("idx"),
        IndexedSeq("idx"),
        "outer",
      ) { case (_, _) => I32(1) })
    }
  }

  @Test def testStreamGroupedMemory(): Unit = {
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(mapIR(StreamGrouped(rangeIR(size), 100))(stream => I32(1)))
    }

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(mapIR(StreamGrouped(rangeIR(size), 100))(stream => sumIR(stream)))
    }
  }

  @Test def testStreamFilterMemory(): Unit =
    assertMemoryDoesNotScaleWithStreamSize(highSize = 100000) { size =>
      StreamLen(filterIR(mapIR(StreamRange(0, size, 1, true))(i => invoke("str", TString, i))) {
        str => invoke("length", TInt32, str) > (size * 9 / 10).toString.size
      })
    }

  @Test def testStreamFlatMapMemory(): Unit = {
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(flatMapIR(filteredRangeStructs(size)) { struct =>
        StreamRange(0, invoke("length", TInt32, GetField(struct, "foo2")), 1, true)
      })
    }

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      sumIR(flatMapIR(filteredRangeStructs(size)) { struct =>
        StreamRange(0, invoke("length", TInt32, GetField(struct, "foo2")), 1, false)
      })
    }
  }

  @Test def testGroupedFlatMapMemManagementMismatch(): Unit =
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldLength(flatMapIR(mapIR(StreamGrouped(rangeStructs(size), 16))(x => ToArray(x))) { a =>
        ToStream(a, false)
      })
    }

  @Test def testStreamTakeWhile(): Unit = {
    val makestream = MakeStream(FastSeq(I32(1), I32(2), I32(0), I32(1), I32(-1)), TStream(TInt32))
    assert(evalStream(takeWhile(makestream)(r => r > 0)) == IndexedSeq(1, 2))
    assert(evalStream(StreamTake(makestream, I32(3))) == IndexedSeq(1, 2, 0))
    assert(evalStream(takeWhile(makestream)(r => NA(TBoolean))) == IndexedSeq())
    assert(evalStream(takeWhile(makestream)(r => If(r > 0, True(), NA(TBoolean)))) == IndexedSeq(
      1,
      2,
    ))
  }

  @Test def testStreamDropWhile(): Unit = {
    val makestream = MakeStream(FastSeq(I32(1), I32(2), I32(0), I32(1), I32(-1)), TStream(TInt32))
    assert(evalStream(dropWhile(makestream)(r => r > 0)) == IndexedSeq(0, 1, -1))
    assert(evalStream(StreamDrop(makestream, I32(3))) == IndexedSeq(1, -1))
    assert(evalStream(dropWhile(makestream)(r => NA(TBoolean))) == IndexedSeq(1, 2, 0, 1, -1))
    assert(evalStream(dropWhile(makestream)(r => If(r > 0, True(), NA(TBoolean)))) == IndexedSeq(
      0,
      1,
      -1,
    ))

  }

  @Test def testStreamTakeDropMemory(): Unit = {
    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldLength(StreamTake(rangeStructs(size), (size / 2).toI))
    }

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldLength(StreamDrop(rangeStructs(size), (size / 2).toI))
    }

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldLength(dropWhile(rangeStructs(size))(elt => GetField(elt, "idx") < (size / 2).toI))
    }

    assertMemoryDoesNotScaleWithStreamSize() { size =>
      foldLength(takeWhile(rangeStructs(size))(elt => GetField(elt, "idx") < (size / 2).toI))
    }
  }

  @Test def testStreamIota(): Unit = {
    assert(evalStream(takeWhile(iota(0, 2))(elt => elt < 10)) == IndexedSeq(0, 2, 4, 6, 8))
    assert(evalStream(StreamTake(iota(5, -5), 3)) == IndexedSeq(5, 0, -5))
  }

  @Test def testStreamIntervalJoin(): Unit = {
    val keyStream = mapIR(StreamRange(0, 9, 1, requiresMemoryManagementPerElement = true)) { i =>
      MakeStruct(FastSeq("i" -> i))
    }
    val kType = TIterable.elementType(keyStream.typ).asInstanceOf[TStruct]
    val rightElemType = TStruct("interval" -> TInterval(TInt32))

    val intervals: IndexedSeq[Interval] =
      for {
        (start, end, includesStart, includesEnd) <- FastSeq(
          (1, 6, true, false),
          (2, 2, false, false),
          (3, 5, true, true),
          (4, 6, true, true),
          (6, 7, false, true),
        )
      } yield Interval(
        IntervalEndpoint(start, if (includesStart) -1 else 1),
        IntervalEndpoint(end, if (includesEnd) 1 else -1),
      )

    val lname = Ref(freshName(), kType)
    val rname = Ref(freshName(), TArray(rightElemType))
    val join =
      ToArray(
        StreamLeftIntervalJoin(
          keyStream,
          ToStream(
            Literal(TArray(rightElemType), intervals.map(Row(_))),
            requiresMemoryManagementPerElement = true,
          ),
          kType.fieldNames.head,
          "interval",
          lname.name,
          rname.name,
          InsertFields(
            lname,
            FastSeq(
              "intervals" -> mapArray(rname) {
                GetField(_, "interval")
              }
            ),
          ),
        )
      )

    assertEvalsTo(
      join,
      FastSeq(
        Row(0, FastSeq()),
        Row(1, FastSeq(intervals(0))),
        Row(2, FastSeq(intervals(0))),
        Row(3, FastSeq(intervals(2), intervals(0))),
        Row(4, FastSeq(intervals(2), intervals(0), intervals(3))),
        Row(5, FastSeq(intervals(2), intervals(0), intervals(3))),
        Row(6, FastSeq(intervals(3))),
        Row(7, FastSeq(intervals(4))),
        Row(8, FastSeq()),
      ),
    )
  }
}
