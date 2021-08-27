package is.hail.io.avro

import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir._
import is.hail.rvd.RVDPartitioner
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalStruct, PStruct}
import is.hail.types.virtual.{TArray, TString, TStruct}
import org.json4s.JValue

class AvroTableReader(partitionReader: AvroPartitionReader, paths: IndexedSeq[String]) extends TableReader {
  def pathsUsed: Seq[String] = paths

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def fullType: TableType = TableType(partitionReader.fullRowType, IndexedSeq(), TStruct())

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
  (partitionReader.rowRequiredness(requestedType.rowType).canonicalPType(requestedType.rowType).asInstanceOf[PStruct],
  PCanonicalStruct(required = true))

  def renderShort(): String = defaultRender()

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val ts = lower(ctx, tr.typ)
    new TableStageIntermediate(ts).asTableValue(ctx)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val partitioner = RVDPartitioner.unkeyed(paths.length)
    val globals = MakeStruct(Seq())
    val contexts = ToStream(Literal(TArray(TString), paths))
    TableStage(
      globals,
      partitioner,
      TableStageDependency.none,
      contexts,
      { ctx =>
        ReadPartition(ctx, requestedType.rowType, partitionReader)
      }
    )
  }
}

object AvroTableReader {
  def fromJValue(jv: JValue): AvroTableReader = {
    implicit val formats = PartitionReader.formats
    val paths = (jv \ "paths").extract[IndexedSeq[String]]
    val partitionReader = (jv \ "partitionReader").extract[AvroPartitionReader]

    new AvroTableReader(partitionReader, paths)
  }
}
