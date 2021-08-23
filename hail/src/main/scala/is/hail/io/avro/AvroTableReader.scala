package is.hail.io.avro

import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.{ExecuteContext, MakeArray, MakeStruct, ReadPartition, Str, TableRead, TableReader, TableStageIntermediate, TableValue, ToStream}
import is.hail.rvd.RVDPartitioner
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalStruct, PStruct}
import is.hail.types.virtual.TStruct

class AvroTableReader(partitionReader: AvroPartitionReader, paths: IndexedSeq[String]) extends TableReader {
  def pathsUsed: Seq[String] = paths

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val partitioner = RVDPartitioner.unkeyed(paths.length)
    val globals = MakeStruct(Seq())
    val contexts = MakeArray(paths.map(Str): _*)
    val ts = TableStage(
      globals,
      partitioner,
      TableStageDependency.none,
      contexts,
      { ctx =>
        ReadPartition(ctx, tr.typ.rowType, partitionReader)
      }
    )
    new TableStageIntermediate(ts).asTableValue(ctx)
  }

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def fullType: TableType = TableType(partitionReader.fullRowType, IndexedSeq(), TStruct())

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
    (partitionReader.rowRequiredness(requestedType.rowType).canonicalPType(requestedType.rowType).asInstanceOf[PStruct],
     PCanonicalStruct(required = true))

  def renderShort(): String = defaultRender()
}
