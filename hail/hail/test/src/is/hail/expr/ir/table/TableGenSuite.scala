package is.hail.expr.ir.table

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{
  Atom, ErrorIDs, GetField, MakeStream, MakeStruct, Ref, Str, StreamRange, TableAggregate,
  TableGetGlobals,
}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{HailException, Interval}

import org.apache.spark.SparkException
import org.junit.jupiter.api.Test
import org.scalatest.matchers.should.Matchers.{intercept => _, _}

class TableGenSuite {

  implicit val execStrategy: Set[ExecStrategy] = ExecStrategy.lowering

  @Test
  def testWithInvalidContextsType(implicit ctx: ExecuteContext): Unit = {
    val ex = intercept[IllegalArgumentException] {
      TypeCheck(ctx, mkTableGen(contexts = Some(Str("oh noes :'("))))
    }

    ex.getMessage should include("contexts")
    ex.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test
  def testWithInvalidGlobalsType(implicit ctx: ExecuteContext): Unit = {
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

  @Test
  def testWithInvalidBodyType(implicit ctx: ExecuteContext): Unit = {
    val ex = intercept[HailException] {
      TypeCheck(ctx, mkTableGen(body = Some((_, _) => Str("oh noes :'("))))
    }
    ex.getCause.getMessage should include("body")
    ex.getCause.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getCause.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test
  def testWithInvalidBodyElementType(implicit ctx: ExecuteContext): Unit = {
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

  @Test
  def testWithInvalidPartitionerKeyType(implicit ctx: ExecuteContext): Unit = {
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

  @Test
  def testWithTooLongPartitionerKeyType(implicit ctx: ExecuteContext): Unit = {
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

  @Test
  def testRequiredness(implicit ctx: ExecuteContext): Unit = {
    val table = mkTableGen()
    val analysis = Requiredness(table, ctx)
    analysis.lookup(table).required shouldBe true
    analysis.states.m.isEmpty shouldBe true
  }

  @Test
  def testLowering(implicit ctx: ExecuteContext): Unit = {
    val rows = collect(mkTableGen())
    assertEvalsTo(rows, RowSeq(FastSeq(0, 0).map(RowSeq(_)), RowSeq(0)))
  }

  @Test
  def testNumberOfContextsMatchesPartitions(implicit ctx: ExecuteContext): Unit = {
    val errorId = 42
    val rows = collect(mkTableGen(
      partitioner = Some(RVDPartitioner.unkeyed(ctx.stateManager, 0)),
      errorId = Some(errorId),
    ))
    val ex = intercept[HailException] {
      loweredExecute(rows, Env.empty, FastSeq(), None)
    }
    ex.errorId shouldBe errorId
    ex.getMessage should include("partitioner contains 0 partitions, got 2 contexts.")
  }

  @Test
  def testRowsAreCorrectlyKeyed(implicit ctx: ExecuteContext): Unit = {
    val errorId = 56
    val rows = collect(mkTableGen(
      partitioner = Some(new RVDPartitioner(
        ctx.stateManager,
        TStruct("a" -> TInt32),
        FastSeq(
          Interval(RowSeq(0), RowSeq(0), true, false),
          Interval(RowSeq(1), RowSeq(1), true, false),
        ),
      )),
      errorId = Some(errorId),
    ))
    val ex = intercept[SparkException] {
      loweredExecute(rows, Env.empty, FastSeq(), None)
    }.getCause.asInstanceOf[HailException]

    ex.errorId shouldBe errorId
    ex.getMessage should include("TableGen: Unexpected key in partition")
  }

  @Test
  def testPruneNoUnusedFields(implicit ctx: ExecuteContext): Unit = {
    val start = mkTableGen()
    val pruned = PruneDeadFields(ctx, start)
    pruned.typ shouldBe start.typ
  }

  @Test
  def testPruneGlobals(implicit ctx: ExecuteContext): Unit = {
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

  @Test
  def testPruneContexts(implicit ctx: ExecuteContext): Unit = {
    val start = mkTableGen()
    val TableGetGlobals(pruned) = PruneDeadFields(ctx, TableGetGlobals(start))
    pruned.typ should not be start.typ
    pruned.typ.rowType shouldBe TStruct()
  }

  def mkTableGen(
    contexts: Option[IR] = None,
    globals: Option[IR] = None,
    body: Option[(Atom, Atom) => IR] = None,
    partitioner: Option[RVDPartitioner] = None,
    errorId: Option[Int] = None,
  )(implicit ctx: ExecuteContext
  ): TableGen =
    tableGen(
      contexts.getOrElse(StreamRange(0, 2, 1)),
      globals.getOrElse(makestruct("g" -> 0)),
      partitioner.getOrElse(RVDPartitioner.unkeyed(ctx.stateManager, 2)),
      errorId.getOrElse(ErrorIDs.NO_ERROR),
    )(
      body.getOrElse((c, g) => MakeStream(makestruct("a" -> c * GetField(g, "g"))))
    )
}
