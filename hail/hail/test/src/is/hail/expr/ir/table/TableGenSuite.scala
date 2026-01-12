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
import org.scalatest.matchers.should.Matchers._
import org.testng.annotations.Test

class TableGenSuite extends HailSuite {

  implicit val execStrategy: Set[ExecStrategy] = ExecStrategy.lowering

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidContextsType(): Unit = {
    val ex = intercept[IllegalArgumentException] {
      TypeCheck(ctx, mkTableGen(contexts = Some(Str("oh noes :'("))))
    }

    ex.getMessage should include("contexts")
    ex.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidGlobalsType(): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(
          globals = Some(Str("oh noes :'(")),
          body = Some((_, _) => MakeStream(IndexedSeq(), TStream(TStruct()))),
        ),
      )
    }
    ex.getCause.getMessage should include("globals")
    ex.getCause.getMessage should include(s"Expected: ${classOf[TStruct].getName}")
    ex.getCause.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidBodyType(): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(ctx, mkTableGen(body = Some((_, _) => Str("oh noes :'("))))
    }
    ex.getCause.getMessage should include("body")
    ex.getCause.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getCause.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidBodyElementType(): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(body =
          Some((_, _) => MakeStream(IndexedSeq(Str("oh noes :'(")), TStream(TString)))
        ),
      )
    }
    ex.getCause.getMessage should include("body.elementType")
    ex.getCause.getMessage should include(s"Expected: ${classOf[TStruct].getName}")
    ex.getCause.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidPartitionerKeyType(): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(partitioner =
          Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32)))
        ),
      )
    }
    ex.getCause.getMessage should include("partitioner")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithTooLongPartitionerKeyType(): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(
        ctx,
        mkTableGen(partitioner =
          Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32)))
        ),
      )
    }
    ex.getCause.getMessage should include("partitioner")
  }

  @Test(groups = Array("requiredness"))
  def testRequiredness(): Unit = {
    val table = mkTableGen()
    val analysis = Requiredness(table, ctx)
    analysis.lookup(table).required shouldBe true
    analysis.states.m.isEmpty shouldBe true
  }

  @Test(groups = Array("lowering"))
  def testLowering(): Unit = {
    val table = collect(mkTableGen())
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, LoweringAnalyses(table, ctx))
    assertEvalsTo(lowered, Row(FastSeq(0, 0).map(Row(_)), Row(0)))
  }

  @Test(groups = Array("lowering"))
  def testNumberOfContextsMatchesPartitions(): Unit = {
    val errorId = 42
    val table = collect(mkTableGen(
      partitioner = Some(RVDPartitioner.unkeyed(ctx.stateManager, 0)),
      errorId = Some(errorId),
    ))
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, LoweringAnalyses(table, ctx))
    val ex = intercept[HailException] {
      loweredExecute(ctx, lowered, Env.empty, FastSeq(), None)
    }
    ex.errorId shouldBe errorId
    ex.getMessage should include("partitioner contains 0 partitions, got 2 contexts.")
  }

  @Test(groups = Array("lowering"))
  def testRowsAreCorrectlyKeyed(): Unit = {
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

    ex.errorId shouldBe errorId
    ex.getMessage should include("TableGen: Unexpected key in partition")
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneNoUnusedFields(): Unit = {
    val start = mkTableGen()
    val pruned = PruneDeadFields(ctx, start)
    pruned.typ shouldBe start.typ
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneGlobals(): Unit = {
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

    pruned.typ should not be start.typ
    pruned.typ.globalType shouldBe TStruct()
    pruned.asInstanceOf[TableGen].globals shouldBe MakeStruct(IndexedSeq())
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneContexts(): Unit = {
    val start = mkTableGen()
    val TableGetGlobals(pruned) = PruneDeadFields(ctx, TableGetGlobals(start))
    pruned.typ should not be start.typ
    pruned.typ.rowType shouldBe TStruct()
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
