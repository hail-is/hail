package is.hail.expr.ir.table

import is.hail.backend.HailStateManager
import is.hail.expr.ir.TestUtils.IRAggCollect
import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.{DArrayLowering, LowerTableIR}
import is.hail.rvd.RVDPartitioner
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq
import org.apache.spark.sql.Row
import org.scalatest.Matchers._
import org.testng.annotations.Test

class TableGenSuite extends HailSuite {

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
      mkTableGen(globals = Some(Str("oh noes :'(")), body = Some(MakeStream(Seq(), TStream(TStruct()))))
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
      mkTableGen(body = Some(MakeStream(Seq(Str("oh noes :'(")), TStream(TString))))
    }
    ex.getMessage should include("body.elementType")
    ex.getMessage should include(s"Expected: ${classOf[TStruct].getName}")
    ex.getMessage should include(s"Actual: ${TString.getClass.getName}")
  }

  @Test(groups = Array("analysis", "requiredness"))
  def testRequiredness: Unit = {
    val table = mkTableGen()
    val analysis = Requiredness(table, ctx)
    analysis.lookup(table).required shouldBe true
    analysis.states.m.isEmpty shouldBe true
  }

  @Test(groups = Array("analysis", "lowering"))
  def testLowering: Unit = {
    val table = TestUtils.collect(mkTableGen())
    val lowered = LowerTableIR(table, DArrayLowering.All, ctx, Analyses(table, ctx), Map.empty)
    assertEvalsTo(lowered, Row(FastIndexedSeq(0, 0).map(Row(_)), Row(0)))
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
      val elem = MakeStruct(Seq("a" -> Ref(cname, TInt32)))
      MakeStream(Seq(elem), TStream(elem.typ))
    })

    val TableAggregate(pruned, _) = PruneDeadFields(ctx,
      TableAggregate(start, IRAggCollect(Ref("row", start.typ.rowType)))
    )

    pruned.typ should not be start.typ
    pruned.typ.globalType shouldBe TStruct()
    pruned.asInstanceOf[TableGen].globals shouldBe MakeStruct(Seq())
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
                 partitioner: Option[RVDPartitioner] = None
                ): TableGen = {
    val theGlobals = globals.getOrElse(MakeStruct(Seq("g" -> 0)))
    val contextName = cname.getOrElse(genUID())
    val globalsName = gname.getOrElse(genUID())

    TableGen(
      contexts.getOrElse(StreamRange(0, 2, 1)),
      theGlobals,
      contextName,
      globalsName,
      body.getOrElse {
        val elem = MakeStruct(Seq(
          "a" -> ApplyBinaryPrimOp(Multiply(), Ref(contextName, TInt32), GetField(theGlobals, "g"))
        ))
        MakeStream(Seq(elem), TStream(elem.typ))
      },
      partitioner.getOrElse(RVDPartitioner.unkeyed(HailStateManager(Map.empty), 2))
    )
  }
}
