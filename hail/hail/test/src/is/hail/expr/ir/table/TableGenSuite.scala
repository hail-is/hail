package is.hail.expr.ir.table

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{
  ApplyBinaryPrimOp, ErrorIDs, GetField, MakeStream, MakeStruct, Ref, Str, StreamRange,
  TableAggregate, TableGetGlobals,
}
import is.hail.expr.ir.lowering.{DArrayLowering, LowerTableIR}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{HailException, Interval}

import org.apache.spark.SparkException
import org.apache.spark.sql.Row

class TableGenSuite extends HailSuite {

  implicit val execStrategy: Set[ExecStrategy] = ExecStrategy.lowering

  test("WithInvalidContextsType") {
    val ex = intercept[IllegalArgumentException] {
      TypeCheck(ctx, mkTableGen(contexts = Some(Str("oh noes :'("))))
    }

    assert(ex.getMessage.contains("contexts"))
    assert(ex.getMessage.contains(s"Expected: ${classOf[TStream].getName}"))
    assert(ex.getMessage.contains(s"Actual: ${TString.getClass.getName}"))
  }

  test("WithInvalidGlobalsType") {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(
          globals = Some(Str("oh noes :'(")),
          body = Some((_, _) => MakeStream(IndexedSeq(), TStream(TStruct()))),
        ),
      )
    }
    assert(ex.getCause.getMessage.contains("globals"))
    assert(ex.getCause.getMessage.contains(s"Expected: ${classOf[TStruct].getName}"))
    assert(ex.getCause.getMessage.contains(s"Actual: ${TString.getClass.getName}"))
  }

  test("WithInvalidBodyType") {
    val ex = intercept[HailException] {
      TypeCheck(ctx, mkTableGen(body = Some((_, _) => Str("oh noes :'("))))
    }
    assert(ex.getCause.getMessage.contains("body"))
    assert(ex.getCause.getMessage.contains(s"Expected: ${classOf[TStream].getName}"))
    assert(ex.getCause.getMessage.contains(s"Actual: ${TString.getClass.getName}"))
  }

  test("WithInvalidBodyElementType") {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(body =
          Some((_, _) => MakeStream(IndexedSeq(Str("oh noes :'(")), TStream(TString)))
        ),
      )
    }
    assert(ex.getCause.getMessage.contains("body.elementType"))
    assert(ex.getCause.getMessage.contains(s"Expected: ${classOf[TStruct].getName}"))
    assert(ex.getCause.getMessage.contains(s"Actual: ${TString.getClass.getName}"))
  }

  test("WithInvalidPartitionerKeyType") {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(partitioner =
          Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32)))
        ),
      )
    }
    assert(ex.getCause.getMessage.contains("partitioner"))
  }

  test("WithTooLongPartitionerKeyType") {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(partitioner =
          Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32)))
        ),
      )
    }
    assert(ex.getCause.getMessage.contains("partitioner"))
  }

  test("Requiredness") {
    val table = mkTableGen()
    val analysis = Requiredness(table, ctx)
    assertEquals(analysis.lookup(table).required, true)
    assertEquals(analysis.states.m.isEmpty, true)
  }

  test("Lowering") {
    val table = collect(mkTableGen())
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, LoweringAnalyses(table, ctx))
    assertEvalsTo(lowered, Row(FastSeq(0, 0).map(Row(_)), Row(0)))
  }

  test("NumberOfContextsMatchesPartitions") {
    val errorId = 42
    val table = collect(mkTableGen(
      partitioner = Some(RVDPartitioner.unkeyed(ctx.stateManager, 0)),
      errorId = Some(errorId),
    ))
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, LoweringAnalyses(table, ctx))
    val ex = intercept[HailException] {
      loweredExecute(ctx, lowered, Env.empty, FastSeq(), None)
    }
    assertEquals(ex.errorId, errorId)
    assert(ex.getMessage.contains("partitioner contains 0 partitions, got 2 contexts."))
  }

  test("RowsAreCorrectlyKeyed") {
    val errorId = 56
    val table = collect(mkTableGen(
      partitioner = Some(new RVDPartitioner(
        ctx.stateManager,
        TStruct("a" -> TInt32),
        FastSeq(
          Interval(Row(0), Row(0), true, false),
          Interval(Row(1), Row(1), true, false),
        ),
      )),
      errorId = Some(errorId),
    ))
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, LoweringAnalyses(table, ctx))
    val ex = intercept[SparkException] {
      loweredExecute(ctx, lowered, Env.empty, FastSeq(), None)
    }.getCause.asInstanceOf[HailException]

    assertEquals(ex.errorId, errorId)
    assert(ex.getMessage.contains("TableGen: Unexpected key in partition"))
  }

  test("PruneNoUnusedFields") {
    val start = mkTableGen()
    val pruned = PruneDeadFields(ctx, start)
    assertEquals(pruned.typ, start.typ)
  }

  test("PruneGlobals") {
    val start = mkTableGen(
      body = Some { (c, _) =>
        val elem = MakeStruct(IndexedSeq("a" -> c))
        MakeStream(IndexedSeq(elem), TStream(elem.typ))
      }
    )

    val TableAggregate(pruned, _) =
      PruneDeadFields(
        ctx,
        TableAggregate(start, IRAggCollect(Ref(TableIR.rowName, start.typ.rowType))),
      )

    assertNotEquals(pruned.typ, start.typ)
    assertEquals(pruned.typ.globalType, TStruct())
    assertEquals(pruned.asInstanceOf[TableGen].globals, MakeStruct(IndexedSeq()))
  }

  test("PruneContexts") {
    val start = mkTableGen()
    val TableGetGlobals(pruned) = PruneDeadFields(ctx, TableGetGlobals(start))
    assertNotEquals(pruned.typ, start.typ)
    assertEquals(pruned.typ.rowType, TStruct())
  }

  def mkTableGen(
    contexts: Option[IR] = None,
    globals: Option[IR] = None,
    body: Option[(Ref, Ref) => IR] = None,
    partitioner: Option[RVDPartitioner] = None,
    errorId: Option[Int] = None,
  ): TableGen = {
    tableGen(
      contexts.getOrElse(StreamRange(0, 2, 1)),
      globals.getOrElse(MakeStruct(IndexedSeq("g" -> 0))),
      partitioner.getOrElse(RVDPartitioner.unkeyed(ctx.stateManager, 2)),
      errorId.getOrElse(ErrorIDs.NO_ERROR),
    )(
      body.getOrElse { (c, g) =>
        val elem = MakeStruct(IndexedSeq(
          "a" -> ApplyBinaryPrimOp(Multiply(), c, GetField(g, "g"))
        ))
        MakeStream(IndexedSeq(elem), TStream(elem.typ))
      }
    )
  }
}
