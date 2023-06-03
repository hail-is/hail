package is.hail.expr.ir.lowering

import cats.implicits.toFoldableOps
import is.hail.expr.ir._
import is.hail.expr.ir.functions.{TableCalculateNewPartitions, TableToValueFunction, WrappedMatrixToTableFunction}
import is.hail.expr.ir.lowering.LowerDistributedSort.LocalSortReader
import is.hail.io.avro.AvroTableReader
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.methods.{ForceCountTable, LocalLDPrune, NPartitionsTable, TableFilterPartitions}
import is.hail.utils.traverseInstanceGenTraversable

import scala.language.higherKinds

object CanLowerEfficiently {
  def apply[M[_]](ir0: BaseIR)(implicit M: MonadLower[M]): M[Either[String, Unit]] = 
    M.ctx.reader { ctx =>
      if (ctx.getFlag("no_whole_stage_codegen") != null) 
        Left("flag 'no_whole_stage_codegen' is enabled")
      else 
        IRTraversal.preOrder(ir0).toTraversable.traverse_ {
          case TableRead(_, _, _: TableNativeReader) |
               TableRead(_, _, _: TableNativeZippedReader) |
               TableRead(_, _, _: StringTableReader) |
               TableRead(_, _, _: MatrixPLINKReader) |
               TableRead(_, _, _: MatrixVCFReader) |
               TableRead(_, _, _: AvroTableReader) |
               TableRead(_, _, _: RVDTableReader) |
               TableRead(_, _, _: LocalSortReader) |
               TableRead(_, _, _: DistributionSortReader) |
               TableRead(_, _, _: MatrixBGENReader) |
               TableRepartition(_, _, RepartitionStrategy.NAIVE_COALESCE) |
               _: TableLiteral |
               _: TableParallelize |
               _: TableRange |
               _: TableKeyBy |
               _: TableOrderBy |
               _: TableFilter |
               _: TableHead |
               _: TableTail |
               _: TableJoin |
               _: TableLeftJoinRightDistinct |
               _: TableMapPartitions |
               _: TableMapRows |
               _: TableMapGlobals |
               _: TableExplode |
               _: TableDistinct |
               _: TableKeyByAndAggregate |
               _: TableAggregateByKey |
               _: TableRename |
               _: TableFilterIntervals |
               _: TableGen |
               _: TableCount |
               _: TableAggregate |
               _: TableCollect |
               _: TableGetGlobals |
               TableToTableApply(_, _: TableFilterPartitions) |
               TableToTableApply(_, WrappedMatrixToTableFunction(_: LocalLDPrune, _, _, _)) |
               TableToValueApply(_, _: ForceCountTable) |
               TableToValueApply(_, _: NPartitionsTable) |
               TableToValueApply(_, _: TableCalculateNewPartitions) =>
            Right()

          case TableRead(_, _, _: TableFromBlockMatrixNativeReader) =>
            Left(s"no lowering for TableFromBlockMatrixNativeReader")

          case _: TableRepartition =>
            Left(s"TableRepartition has no lowered implementation")

          case TableIntervalJoin(_, _, _, product) =>
            if (product) Left("TableIntervalJoin with \"product=true\" has no lowered implementation")
            else Right()

          case t: TableUnion =>
            if (t.children.length > 16) Left(s"TableUnion lowering generates deeply nested IR if it has many children")
            else Right()

          case t: TableMultiWayZipJoin =>
            Left(s"TableMultiWayZipJoin is not passing tests due to problems in ptype inference in StreamZipJoin")

          case TableToValueApply(_, f: TableToValueFunction) =>
            Left(s"TableToValueApply: no lowering for ${f.getClass.getName}")

          case _: TableToTableApply =>
            Left(s"TableToTableApply")

          case _: BlockMatrixToTableApply => Left(s"BlockMatrixToTableApply")
          case _: BlockMatrixToTable => Left(s"BlockMatrixToTable has no lowered implementation")
          case _: BlockMatrixAgg => Left(s"BlockMatrixAgg needs to do tree aggregation")
          case _: BlockMatrixIR => Left(s"BlockMatrixIR lowering not yet efficient/scalable")
          case _: BlockMatrixWrite => Left(s"BlockMatrixIR lowering not yet efficient/scalable")
          case _: BlockMatrixMultiWrite => Left(s"BlockMatrixIR lowering not yet efficient/scalable")
          case _: BlockMatrixCollect => Left(s"BlockMatrixIR lowering not yet efficient/scalable")
          case _: BlockMatrixToValueApply => Left(s"BlockMatrixIR lowering not yet efficient/scalable")

          case _: MatrixMultiWrite => Left(s"no lowering for MatrixMultiWrite")

          case TableWrite(_, writer) =>
            if (!writer.canLowerEfficiently) Left(s"writer has no efficient lowering: ${writer.getClass.getSimpleName}")
            else Right()

          case TableMultiWrite(_, _) =>
            Left(s"no lowering available for TableMultiWrite")

          case RelationalRef(_, _) =>
            throw new RuntimeException(s"unexpected relational ref")

          case _: ApplySeeded =>
            Left("seeded randomness does not satisfy determinism restrictions in lowered IR")

          case x: IR if !x.children.forall(_.isInstanceOf[IR]) =>
            // nodes with relational children should be enumerated above explicitly
            throw new RuntimeException(s"IR must be enumerated explicitly: ${x.getClass.getName}")
        }
  }
}
