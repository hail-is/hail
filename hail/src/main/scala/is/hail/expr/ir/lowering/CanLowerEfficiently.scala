package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.ir.functions.TableToValueFunction
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.gen.MatrixGENReader
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
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
        case TableRead(_, _, _: TableNativeReader) =>
        case TableRead(_, _, _: TableNativeZippedReader) =>
        case TableRead(_, _, _: TextTableReader) =>
        case TableRead(_, _, _: MatrixPLINKReader) =>
        case TableRead(_, _, _: MatrixVCFReader) =>
        case TableRead(_, _, _: MatrixBGENReader) =>
          fail(s"no lowering for MatrixBGENReader")
        case TableRead(_, _, _: MatrixGENReader) =>
          fail(s"no lowering for MatrixGENReader")
        case TableRead(_, _, _: TableFromBlockMatrixNativeReader) =>
          fail(s"no lowering for TableFromBlockMatrixNativeReader")

        case t: TableLiteral =>
        case t: TableRepartition => fail(s"TableRepartition has no lowered implementation")
        case t: TableParallelize =>
        case t: TableRange =>
        case t: TableKeyBy =>
        case t: TableFilter =>
        case t: TableHead => fail("TableHead has no short-circuit using known partition counts")
        case t: TableTail => fail("TableTail has no short-circuit using known partition counts")
        case t: TableJoin if t.joinType == "inner" =>
          fail("TableJoin with inner join generates a stream that iterates over the entire left stream (no early truncation)")
        case t: TableJoin =>
        case t: TableIntervalJoin => fail(s"TableIntervalJoin has no lowered implementation")
        case t: TableMultiWayZipJoin =>
        case t: TableLeftJoinRightDistinct =>
        case t: TableMapPartitions =>
        case t: TableMapRows => if (ContainsScan(t.newRow)) fail("TableMapRows does not have a scalable implementation of scans")
        case t: TableMapGlobals =>
        case t: TableExplode =>
        case t: TableUnion if t.children.length > 16 => fail(s"TableUnion lowering generates deeply nested IR if it has many children")
        case t: TableUnion =>
        case t: TableMultiWayZipJoin => fail(s"TableMultiWayZipJoin is not passing tests due to problems in ptype inference in StreamZipJoin")
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
        case x: BlockMatrixWrite => fail(s"BlockMatrixIR lowering not yet efficient/scalable")
        case x: BlockMatrixMultiWrite => fail(s"BlockMatrixIR lowering not yet efficient/scalable")

        case mmr: MatrixMultiWrite => fail(s"no lowering for MatrixMultiWrite")

        case TableCount(_) =>
        case TableToValueApply(_, ForceCountTable()) =>
        case TableToValueApply(_, NPartitionsTable()) =>
        case TableToValueApply(_, f: TableToValueFunction) => fail(s"TableToValueApply: no lowering for ${ f.getClass.getName }")
        case TableAggregate(_, _) => fail("TableAggregate needs a tree aggregate implementation to scale")
        case TableCollect(_) =>
        case TableGetGlobals(_) =>
        case TableWrite(_, _) => fail(s"table writers don't correctly generate indices")

        case RelationalRef(_, _) => throw new RuntimeException(s"unexpected relational ref")

        case x: IR =>
          // nodes with relational children should be enumerated above explicitly
          if (!x.children.forall(_.isInstanceOf[IR])) {
            throw new RuntimeException(s"IR must be enumerated explicitly: ${ x.getClass.getName }")
          }
      }

      if (prohibitiveReason.isEmpty)
        ir.children.foreach(recur)
    }

    recur(ir0)
    prohibitiveReason
  }
}