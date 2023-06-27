package is.hail.expr.ir.table

import is.hail.TestUtils.loweredExecute
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.TestUtils.IRAggCollect
import is.hail.expr.ir._
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, HailException, Interval}
import is.hail.{ExecStrategy, HailSuite, MonadRunSupport}
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.scalatest.Matchers._
import org.testng.annotations.Test

class TableGenSuite extends HailSuite with MonadRunSupport {

  implicit val execStrategy = ExecStrategy.lowering

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidContextsType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(contexts = Some(Str("oh noes :'(")))
    }

    ex.getMessage should include("contexts")
    ex.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidGlobalsType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(globals = Some(Str("oh noes :'(")), body = Some(MakeStream(IndexedSeq(), TStream(TStruct()))))
    }
    ex.getMessage should include("globals")
    ex.getMessage should include(s"Expected: ${classOf[TStruct].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidBodyType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(body = Some(Str("oh noes :'(")))
    }
    ex.getMessage should include("body")
    ex.getMessage should include(s"Expected: ${classOf[TStream].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidBodyElementType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(body = Some(MakeStream(IndexedSeq(Str("oh noes :'(")), TStream(TString))))
    }
    ex.getMessage should include("body.elementType")
    ex.getMessage should include(s"Expected: ${classOf[TStruct].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithInvalidPartitionerKeyType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(partitioner = Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32))))
    }
    ex.getMessage should include("partitioner")
  }

  @Test(groups = Array("construction", "typecheck"))
  def testWithTooLongPartitionerKeyType: Unit = {
    val ex = intercept[IllegalArgumentException] {
      mkTableGen(partitioner = Some(RVDPartitioner.empty(ctx.stateManager, TStruct("does-not-exist" -> TInt32))))
    }
    ex.getMessage should include("partitioner")
  }

  @Test(groups = Array("requiredness"))
  def testRequiredness: Unit = {
    val table = mkTableGen()
    val analysis = Requiredness(table).apply(ctx)
    analysis.lookup(table).required shouldBe true
    analysis.states.m.isEmpty shouldBe true
  }

  @Test(groups = Array("lowering"))
  def testLowering: Unit = {
    val table = TestUtils.collect(mkTableGen())
    assertEvalsTo(table, Row(FastIndexedSeq(0, 0).map(Row(_)), Row(0)))
  }

  @Test(groups = Array("lowering"))
  def testNumberOfContextsMatchesPartitions: Unit = {
    val errorId = 42
    val table = TestUtils.collect(mkTableGen(
      partitioner = Some(RVDPartitioner.unkeyed(ctx.stateManager, 0)),
      errorId = Some(errorId)
    ))
    val ex = intercept[HailException] {
      ExecuteContext.scoped() { ctx =>
        loweredExecute(ctx, table, Env.empty, FastIndexedSeq(), None)
      }
    }
    ex.errorId shouldBe errorId
    ex.getMessage should include("partitioner contains 0 partitions, got 2 contexts.")
  }

  @Test(groups = Array("lowering"))
  def testRowsAreCorrectlyKeyed: Unit = {
    val errorId = 56
    val table = TestUtils.collect(mkTableGen(
      partitioner = Some(new RVDPartitioner(ctx.stateManager, TStruct("a" -> TInt32), FastIndexedSeq(
        Interval(Row(0), Row(0), true, false), Interval(Row(1), Row(1), true, false)
      ))),
      errorId = Some(errorId)
    ))
    val ex = intercept[SparkException] {
      ExecuteContext.scoped() { ctx =>
        loweredExecute(ctx, table, Env.empty, FastIndexedSeq(), None)
      }
    }.getCause.asInstanceOf[HailException]

    ex.errorId shouldBe errorId
    ex.getMessage should include("TableGen: Unexpected key in partition")
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneNoUnusedFields: Unit = {
    val start = mkTableGen()
    val pruned = PruneDeadFields(ctx, start)
    pruned.typ shouldBe start.typ
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneGlobals: Unit = {
    val cname = "contexts"
    val start = mkTableGen(cname = Some(cname), body = Some {
      val elem = MakeStruct(IndexedSeq("a" -> Ref(cname, TInt32)))
      MakeStream(IndexedSeq(elem), TStream(elem.typ))
    })

    val TableAggregate(pruned, _) = PruneDeadFields(ctx,
      TableAggregate(start, IRAggCollect(Ref("row", start.typ.rowType)))
    )

    pruned.typ should not be start.typ
    pruned.typ.globalType shouldBe TStruct()
    pruned.asInstanceOf[TableGen].globals shouldBe MakeStruct(IndexedSeq())
  }

  @Test(groups = Array("optimization", "prune"))
  def testPruneContexts: Unit = {
    val start = mkTableGen()
    val TableGetGlobals(pruned) = PruneDeadFields(ctx, TableGetGlobals(start))
    pruned.typ should not be start.typ
    pruned.typ.rowType shouldBe TStruct()
  }

  def mkTableGen(contexts: Option[IR] = None,
                 globals: Option[IR] = None,
                 cname: Option[String] = None,
                 gname: Option[String] = None,
                 body: Option[IR] = None,
                 partitioner: Option[RVDPartitioner] = None,
                 errorId: Option[Int] = None
                ): TableGen = {
    val theGlobals = globals.getOrElse(MakeStruct(IndexedSeq("g" -> 0)))
    val contextName = cname.getOrElse(genUID())
    val globalsName = gname.getOrElse(genUID())

    TableGen(
      contexts.getOrElse(StreamRange(0, 2, 1)),
      theGlobals,
      contextName,
      globalsName,
      body.getOrElse {
        val elem = MakeStruct(IndexedSeq(
          "a" -> ApplyBinaryPrimOp(Multiply(), Ref(contextName, TInt32), GetField(theGlobals, "g"))
        ))
        MakeStream(IndexedSeq(elem), TStream(elem.typ))
      },
      partitioner.getOrElse(RVDPartitioner.unkeyed(ctx.stateManager, 2)),
      errorId.getOrElse(ErrorIDs.NO_ERROR)
    )
  }
}
