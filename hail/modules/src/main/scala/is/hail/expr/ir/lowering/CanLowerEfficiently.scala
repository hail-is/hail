package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.functions.{
  TableCalculateNewPartitions, TableToValueFunction, WrappedMatrixToTableFunction,
}
import is.hail.expr.ir.lowering.LowerDistributedSort.LocalSortReader
import is.hail.io.avro.AvroTableReader
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.methods.{ForceCountTable, LocalLDPrune, NPartitionsTable, TableFilterPartitions}

object CanLowerEfficiently {
  def apply(ctx: ExecuteContext, ir0: BaseIR): Option[String] = {

    if (ctx.getFlag("no_whole_stage_codegen") != null)
      return Some("flag 'no_whole_stage_codegen' is enabled")

    var prohibitiveReason: Option[String] = None

    def fail(reason: String): Unit = {
      assert(prohibitiveReason.isEmpty)
      prohibitiveReason = Some(reason)
    }

    def recur(ir: BaseIR): Unit = {

      ir match {
        case TableRead(_, _, _: TableNativeReader) =>
        case TableRead(_, _, _: TableNativeZippedReader) =>
        case TableRead(_, _, _: StringTableReader) =>
        case TableRead(_, _, _: MatrixPLINKReader) =>
        case TableRead(_, _, _: MatrixVCFReader) =>
        case TableRead(_, _, _: AvroTableReader) =>
        case TableRead(_, _, _: RVDTableReader) =>
        case TableRead(_, _, _: LocalSortReader) =>
        case TableRead(_, _, _: DistributionSortReader) =>
        case TableRead(_, _, _: MatrixBGENReader) =>
        case TableRead(_, _, _: TableFromBlockMatrixNativeReader) =>
          fail(s"no lowering for TableFromBlockMatrixNativeReader")

        case _: TableLiteral =>
        case TableRepartition(_, _, RepartitionStrategy.NAIVE_COALESCE) =>
        case _: TableRepartition => fail(s"TableRepartition has no lowered implementation")
        case _: TableParallelize =>
        case _: TableRange =>
        case TableKeyBy(_, _, _) =>
        case _: TableOrderBy =>
        case _: TableFilter =>
        case _: TableHead =>
        case _: TableTail =>
        case _: TableJoin =>
        case _: TableIntervalJoin =>
        case _: TableLeftJoinRightDistinct =>
        case _: TableMapPartitions =>
        case _: TableMapRows =>
        case _: TableMapGlobals =>
        case _: TableExplode =>
        case t: TableUnion if t.childrenSeq.length > 16 =>
          fail(s"TableUnion lowering generates deeply nested IR if it has many children")
        case _: TableUnion =>
        case _: TableMultiWayZipJoin => fail(
            s"TableMultiWayZipJoin is not passing tests due to problems in ptype inference in StreamZipJoin"
          )
        case _: TableDistinct =>
        case _: TableKeyByAndAggregate =>
        case _: TableAggregateByKey =>
        case _: TableRename =>
        case _: TableFilterIntervals =>
        case _: TableGen =>
        case TableToTableApply(_, TableFilterPartitions(_, _)) =>
        case TableToTableApply(_, WrappedMatrixToTableFunction(_: LocalLDPrune, _, _, _)) =>
        case _: TableToTableApply => fail(s"TableToTableApply")
        case _: BlockMatrixToTableApply => fail(s"BlockMatrixToTableApply")
        case _: BlockMatrixToTable => fail(s"BlockMatrixToTable has no lowered implementation")

        case _: BlockMatrixAgg => fail(s"BlockMatrixAgg needs to do tree aggregation")
        case _: BlockMatrixIR => fail(s"BlockMatrixIR lowering not yet efficient/scalable")
        case _: BlockMatrixWrite => fail(s"BlockMatrixIR lowering not yet efficient/scalable")
        case _: BlockMatrixMultiWrite => fail(s"BlockMatrixIR lowering not yet efficient/scalable")
        case _: BlockMatrixCollect => fail(s"BlockMatrixIR lowering not yet efficient/scalable")
        case _: BlockMatrixToValueApply =>
          fail(s"BlockMatrixIR lowering not yet efficient/scalable")

        case _: MatrixMultiWrite =>

        case TableCount(_) =>
        case TableToValueApply(_, ForceCountTable()) =>
        case TableToValueApply(_, NPartitionsTable()) =>
        case TableToValueApply(_, TableCalculateNewPartitions(_)) =>
        case TableToValueApply(_, f: TableToValueFunction) =>
          fail(s"TableToValueApply: no lowering for ${f.getClass.getName}")
        case TableAggregate(_, _) =>
        case TableCollect(_) =>
        case TableGetGlobals(_) =>

        case TableWrite(_, writer) => if (!writer.canLowerEfficiently)
            fail(s"writer has no efficient lowering: ${writer.getClass.getSimpleName}")
        case TableMultiWrite(_, _) =>

        case RelationalRef(_, _) => throw new RuntimeException(s"unexpected relational ref")

        case x: IR =>
          // nodes with relational children should be enumerated above explicitly
          if (!x.children.forall(_.isInstanceOf[IR])) {
            throw new RuntimeException(s"IR must be enumerated explicitly: ${x.getClass.getName}")
          }
      }

      ir.children.foreach { child =>
        if (prohibitiveReason.isEmpty)
          recur(child)
      }
    }

    recur(ir0)
    prohibitiveReason
  }
}
