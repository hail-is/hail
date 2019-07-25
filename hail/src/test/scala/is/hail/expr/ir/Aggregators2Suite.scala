package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class Aggregators2Suite extends HailSuite {

  def assertAggEquals(
    aggSig: AggSignature2,
    initArgs: IndexedSeq[IR],
    seqArgs: IndexedSeq[IndexedSeq[IR]],
    expected: Any,
    args: IndexedSeq[(String, (Type, Any))] = FastIndexedSeq(),
    nPartitions: Int = 2): Unit = {
    assert(seqArgs.length >= 2 * nPartitions, s"Test aggregators with a larger stream!")

    val argT = PType.canonical(TStruct(args.map { case (n, (typ, _)) => n -> typ }: _*)).asInstanceOf[PStruct]
    val argVs = Row.fromSeq(args.map { case (_, (_, v)) => v })
    val argRef = Ref(genUID(), argT.virtualType)
    val spec = CodecSpec.defaultUncompressed

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
    assert(resultType.virtualType.typeCheck(expected))

    Region.scoped { region =>
      val argOff = ScalaToRegionValue(region, argT, argVs)
      val serializedParts = seqArgs.grouped(math.ceil(seqArgs.length / nPartitions.toDouble).toInt).map { seqs =>
        val partitionOp = Begin(
          InitOp2(0, initArgs, aggSig) +:
            seqs.map { s => SeqOp2(0, s, aggSig) } :+
            SerializeAggs(0, 0, spec, Array(aggSig)))

        val (_, f) = CompileWithAggregators2[Long, Unit](
          Array(aggSig),
          argRef.name, argRef.pType,
          args.map(_._1).foldLeft[IR](partitionOp) { case (op, name) =>
              Let(name, GetField(argRef, name), op)
          })

        val initAndSeq = f(0, region)
        Region.smallScoped { aggRegion =>
          initAndSeq.newAggState(aggRegion)
          initAndSeq(region, argOff, false)
          initAndSeq.getSerializedAgg(0)
        }
      }

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
        assert(double.get(0) == double.get(1)) // state does not change through serialization
        assert(double.get(0) == expected)
      }
    }
  }

  val t = TStruct("a" -> TString(), "b" -> TInt64())
  val rows = FastIndexedSeq(Row("abcd", 5L), null, Row(null, -2L), Row("abcd", 7L), null, Row("foo", null))
  val arrayType = TArray(t)

  val pnnAggSig = AggSignature2(PrevNonnull(), FastSeq[Type](), FastSeq[Type](t), None)
  val countAggSig = AggSignature2(Count(), FastSeq[Type](), FastSeq[Type](), None)
  val sumAggSig = AggSignature2(Sum(), FastSeq[Type](), FastSeq[Type](TInt64()), None)

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

  @Test def testPrevNonnull() {
    val aggSig = AggSignature2(PrevNonnull(), FastSeq(), FastSeq(t), None)
    val seqOpArgs = Array.tabulate(rows.length)(i => FastIndexedSeq[IR](ArrayRef(Ref("rows", TArray(t)), i)))

    assertAggEquals(aggSig, FastIndexedSeq(), seqOpArgs, expected = rows.last, args = FastIndexedSeq(("rows", (arrayType, rows))))
  }

  @Test def testArrayElementsAgg() {
    val aggSigs = FastIndexedSeq(pnnAggSig, countAggSig, sumAggSig)
    val lcAggSig = AggSignature2(AggElementsLengthCheck(), FastSeq[Type](TVoid), FastSeq[Type](TInt32()), Some(aggSigs))
    val eltAggSig = AggSignature2(AggElements(), FastSeq[Type](), FastSeq[Type](TInt32(), TVoid), Some(aggSigs))


    val value = FastIndexedSeq(
      FastIndexedSeq(Row("a", 0L), Row("b", 0L), Row("c", 0L), Row("f", 0L)),
      FastIndexedSeq(Row("a", 1L), null, Row("c", 1L), null),
      FastIndexedSeq(Row("a", 2L), Row("b", 2L), null, Row("f", 2L)),
      FastIndexedSeq(Row("a", 3L), Row("b", 3L), Row("c", 3L), Row("f", 3L)),
      FastIndexedSeq(Row("a", 4L), Row("b", 4L), Row("c", 4L), null),
      FastIndexedSeq(null, null, null, Row("f", 5L)))

    val expected =
      FastIndexedSeq(
        Row(Row("a", 4), 6L, 10L),
        Row(Row("b", 4), 6L, 9L),
        Row(Row("c", 4), 6L, 8L),
        Row(Row("f", 5), 6L, 10L))

    val arrayPType = PType.canonical(arrayType).asInstanceOf[PArray]

    val array = Ref("array", arrayType)
    val stream = Ref("stream", TArray(arrayType))
    val idx = Ref("idx", TInt32())
    val elt = Ref("elt", t)

    val spec = CodecSpec.defaultUncompressed
    val partitioned = value.grouped(3).toFastIndexedSeq

    val (_, initAndSeqF) = CompileWithAggregators2[Long, Unit](
      Array(lcAggSig),
      "stream", stream.pType,
      Begin(FastIndexedSeq(
        InitOp2(0, FastIndexedSeq(Begin(FastIndexedSeq[IR](
          InitOp2(0, FastIndexedSeq(), pnnAggSig),
          InitOp2(1, FastIndexedSeq(), countAggSig),
          InitOp2(2, FastIndexedSeq(), sumAggSig)
        ))), lcAggSig),
        ArrayFor(stream,
          array.name,
          Begin(FastIndexedSeq(
            SeqOp2(0, FastIndexedSeq(ArrayLen(array)), lcAggSig),
            ArrayFor(
              ArrayRange(I32(0), ArrayLen(array), I32(1)),
              idx.name,
              Let(elt.name,
                ArrayRef(array, idx),
                SeqOp2(0,
                  FastIndexedSeq(idx,
                    Begin(FastIndexedSeq(
                      SeqOp2(0, FastIndexedSeq(elt), pnnAggSig),
                      SeqOp2(1, FastIndexedSeq(), countAggSig),
                      SeqOp2(2, FastIndexedSeq(GetField(elt, "b")), sumAggSig)))),
                  eltAggSig)))))),
        SerializeAggs(0, 0, spec, FastIndexedSeq(lcAggSig)))))

    val (rt: PTuple, resultF) = CompileWithAggregators2[Long](
      Array(lcAggSig, lcAggSig), ResultOp2(0, FastIndexedSeq(lcAggSig)))

    val aggs = Region.scoped { region =>
      val f = initAndSeqF(0, region)

      partitioned.map { case lit =>
        val voff = ScalaToRegionValue(region, arrayPType, lit)

        Region.scoped { aggRegion =>
          f.newAggState(aggRegion)
          f(region, voff, false)
          f.getSerializedAgg(0)
        }
      }
    }

    val (_, deserializeAndComb) = CompileWithAggregators2[Unit](
      Array(lcAggSig, lcAggSig),
      Begin(
        DeserializeAggs(0, 0, spec, FastIndexedSeq(lcAggSig)) +:
          Array.range(1, aggs.length).flatMap { i =>
            FastIndexedSeq(
              DeserializeAggs(1, i, spec, FastIndexedSeq(lcAggSig)),
              CombOp2(0, 1, lcAggSig))
          }))

    Region.scoped { region =>
      val comb = deserializeAndComb(0, region)
      val resF = resultF(0, region)

      Region.scoped { aggRegion =>
        comb.newAggState(aggRegion)
        aggs.zipWithIndex.foreach { case (agg, i) =>
          comb.setSerializedAgg(i, agg)
        }
        comb(region)
        resF.setAggState(aggRegion, comb.getAggOffset())
        val res = resF(region)

        assert(SafeRow(rt, region, res).get(0) == expected)
      }
    }
  }
}
