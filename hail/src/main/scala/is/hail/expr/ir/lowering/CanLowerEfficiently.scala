package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.methods.{ForceCountTable, NPartitionsTable}

object CanLowerEfficiently {
  def apply(ir0: BaseIR): Option[String] = {

    var prohibitiveReason: Option[String] = None

    def fail(reason: String): Unit = {
      assert(prohibitiveReason.isEmpty)
      prohibitiveReason = Some(reason)
    }

    def recur(ir: BaseIR): Unit = {

      ir match {
        case t: TableRead =>
        case t: TableLiteral =>
        case t: TableRepartition => fail(s"TableRepartition has no lowered implementation")
        case t: TableParallelize =>
        case t: TableRange =>
        case t: TableKeyBy =>
        case t: TableFilter =>
        case t: TableHead => fail("TableHead has no short-circuit using known partition counts")
        case t: TableTail => fail("TableTail has no short-circuit using known partition counts")
        case t: TableJoin =>
        case t: TableIntervalJoin => fail(s"TableIntervalJoin has no lowered implementation")
        case t: TableMultiWayZipJoin =>
        case t: TableLeftJoinRightDistinct =>
        case t: TableMapPartitions =>
        case t: TableMapRows => if (ContainsScan(t.newRow)) fail("TableMapRows does not have a scalable implementation of scans")
        case t: TableMapGlobals =>
        case t: TableExplode =>
        case t: TableUnion =>
        case t: TableDistinct =>
        case t: TableKeyByAndAggregate => fail("TableKeyByAndAggregate has no map-side combine")
        case t: TableAggregateByKey =>
        case t: TableOrderBy =>
        case t: TableRename =>
        case t: TableFilterIntervals => fail(s"TableFilterIntervals does a linear scan")
        case t: TableToTableApply => fail(s"TableToTableApply")
        case t: BlockMatrixToTableApply => fail(s"BlockMatrixToTableApply")
        case t: BlockMatrixToTable => fail(s"BlockMatrixToTable has no lowered implementation")

        case x: BlockMatrixIR => fail(s"BlockMatrixIR lowering not yet efficient/scalable")

        case TableCount(_) =>
        case TableToValueApply(_, ForceCountTable()) =>
        case TableToValueApply(_, NPartitionsTable()) =>
        case TableAggregate(_, _) => fail("TableAggregate needs a tree aggregate implementation to scale")
        case TableCollect(_) =>
        case TableGetGlobals(_) =>
        case TableWrite(_, _) => fail(s"table writers don't correctly generate indices")

        case RelationalRef(_, _) => throw new RuntimeException(s"unexpected relational ref")

        case x: IR =>
          // nodes with relational children should be enumerated above explicitly
          assert(x.children.forall(_.isInstanceOf[IR]))
      }

      if (prohibitiveReason.isEmpty)
        ir.children.foreach(recur)
    }

    recur(ir0)
    prohibitiveReason
  }
}