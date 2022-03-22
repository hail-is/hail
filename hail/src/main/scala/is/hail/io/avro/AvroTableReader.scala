package is.hail.io.avro

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir._
import is.hail.rvd.RVDPartitioner
import is.hail.types.TableType
import is.hail.types.physical.{PCanonicalStruct, PStruct}
import is.hail.types.virtual.{TArray, TString, TStruct}
import is.hail.utils.plural
import org.apache.spark.sql.Row
import org.json4s.{Formats, JValue}

class AvroTableReader(
  partitionReader: AvroPartitionReader,
  paths: IndexedSeq[String],
  unsafeOptions: Option[UnsafeAvroTableReaderOptions] = None
) extends TableReader {

  private val partitioner: RVDPartitioner = unsafeOptions.map { case UnsafeAvroTableReaderOptions(key, intervals, _) =>
    require(intervals.length == paths.length,
      s"There must be one partition interval per avro file, have ${paths.length} ${plural(paths.length, "file")} and ${intervals.length} ${plural(intervals.length, "interval")}")
    RVDPartitioner.generate(partitionReader.fullRowType.typeAfterSelectNames(key), intervals)
  }.getOrElse {
    RVDPartitioner.unkeyed(paths.length)
  }

  def pathsUsed: Seq[String] = paths

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def fullType: TableType =
    TableType(partitionReader.fullRowType, unsafeOptions.map(_.key).getOrElse(IndexedSeq()), TStruct())

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) =
  (partitionReader.rowRequiredness(requestedType.rowType).canonicalPType(requestedType.rowType).asInstanceOf[PStruct],
  PCanonicalStruct(required = true))

  def renderShort(): String = defaultRender()

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val ts = lower(ctx, tr.typ)
    new TableStageIntermediate(ts).asTableValue(ctx)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
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
    implicit val formats: Formats = PartitionReader.formats + new UnsafeAvroTableReaderOptionsSerializer
    val paths = (jv \ "paths").extract[IndexedSeq[String]]
    val partitionReader = (jv \ "partitionReader").extract[AvroPartitionReader]
    val unsafeOptions = (jv \ "unsafeOptions").extract[Option[UnsafeAvroTableReaderOptions]]

    new AvroTableReader(partitionReader, paths, unsafeOptions)
  }
}
