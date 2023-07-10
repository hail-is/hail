package is.hail.io.avro

import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage, TableStageDependency}
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical.{PCanonicalStruct, PCanonicalTuple, PInt64Required}
import is.hail.types.virtual._
import is.hail.types.{TableType, VirtualTypeWithReq}
import is.hail.utils.{FastIndexedSeq, plural}
import org.json4s.{Formats, JValue}

class AvroTableReader(
  partitionReader: AvroPartitionReader,
  paths: IndexedSeq[String],
  unsafeOptions: Option[UnsafeAvroTableReaderOptions] = None
) extends TableReaderWithExtraUID {

  private val partitioner: RVDPartitioner = unsafeOptions.map { case UnsafeAvroTableReaderOptions(key, intervals, _) =>
    require(intervals.length == paths.length,
      s"There must be one partition interval per avro file, have ${paths.length} ${plural(paths.length, "file")} and ${intervals.length} ${plural(intervals.length, "interval")}")
    RVDPartitioner.generate(null, partitionReader.fullRowType.typeAfterSelectNames(key), intervals)
  }.getOrElse {
    RVDPartitioner.unkeyed(null, paths.length)
  }

  def pathsUsed: Seq[String] = paths

  def partitionCounts: Option[IndexedSeq[Long]] = None

  override def uidType = TTuple(TInt64, TInt64)

  override def fullTypeWithoutUIDs: TableType =
    TableType(partitionReader.fullRowTypeWithoutUIDs, unsafeOptions.map(_.key).getOrElse(IndexedSeq()), TStruct())

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(requestedType.rowType, partitionReader.rowRequiredness(requestedType.rowType))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct(required = true))

  def renderShort(): String = defaultRender()

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globals = MakeStruct(FastIndexedSeq())
    val contexts = zip2(ToStream(Literal(TArray(TString), paths)), StreamIota(I32(0), I32(1)), ArrayZipBehavior.TakeMinLength) { (path, idx) =>
      MakeStruct(Array("partitionPath" -> path, "partitionIndex" -> Cast(idx, TInt64)))
    }
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

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    throw new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lowerGlobals not implemented")
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
