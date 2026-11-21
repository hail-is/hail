package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{GetField, ReadPartition}
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.FS
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{fatal, Interval}
import is.hail.variant.ReferenceGenomeEnv

import org.json4s.{DefaultFormats, Formats, JObject, JString, JValue}

object TableColumnarReader {

  /** A reader of the columnar table at `path`. Its reference genomes must be equal to any of the
    * same name in `references`.
    */
  def apply(fs: FS, references: ReferenceGenomeEnv, path: String): TableColumnarReader =
    new TableColumnarReader(dataset(fs, references, path), None)

  private def dataset(fs: FS, references: ReferenceGenomeEnv, path: String): ColumnarDataset = {
    val manifest = ColumnarManifest.read(fs, path)
    if (manifest.matrix.isDefined)
      fatal(s"columnar table '$path': it is a matrix table, not a table")
    new ColumnarDataset(
      path,
      manifest,
      references.withReferences(manifest.referenceGenomes),
      TableReader.uidFieldName,
    )
  }

  /** A reader from its JSON: its `path`, and a key-range read, if any (see
    * [[ColumnarReadIntervals.fromJValue]]).
    */
  def fromJValue(fs: FS, references: ReferenceGenomeEnv, jv: JValue): TableColumnarReader = {
    implicit val formats: Formats = DefaultFormats
    val path = (jv \ "path").extract[String]
    val dataset = this.dataset(fs, references, path)
    new TableColumnarReader(
      dataset,
      ColumnarReadIntervals.fromJValue(jv, dataset.references, dataset.keyType, path),
    )
  }
}

/** Reads a table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`).
  *
  * The key, partitioning and distinct keying come from the manifest, so the table is never
  * re-sorted. Row partitions with no rows are dropped. The row UID is the Row ID.
  *
  * A key-range read of `intervals` reads only the rows whose key is in them, and only the row
  * partitions whose bounds overlap them (see [[ColumnarReadIntervals]]).
  */
class TableColumnarReader private (
  dataset: ColumnarDataset,
  val intervals: Option[ColumnarReadIntervals],
) extends TableReaderWithExtraUID with ColumnarKeyRangeReader {

  def path: String = dataset.path

  override def pathsUsed: Seq[String] = FastSeq(path)

  override def restrictedTo(intervals: IndexedSeq[Interval]): Option[TableColumnarReader] =
    ColumnarReadIntervals.restrict(dataset.keyType, this.intervals, intervals).map(r =>
      new TableColumnarReader(dataset, Some(r))
    )

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def isDistinctlyKeyed: Boolean = dataset.manifest.distinctlyKeyed

  override def uidType: Type = TInt64

  override def fullTypeWithoutUIDs: TableType =
    TableType(dataset.rowType, dataset.manifest.key, dataset.globalsPType.virtualType)

  override protected def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(dataset.rowsReader.rowPType(requestedType.rowType))

  override protected def uidRequiredness: VirtualTypeWithReq = VirtualTypeWithReq(PInt64(true))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(dataset.globalsRequiredness(requestedType.globalType))

  override def toJValue: JValue =
    JObject("name" -> JString("TableColumnarReader"), "path" -> JString(path))
      .merge(ColumnarReadIntervals.toJValue(dataset.keyType, intervals))

  override def renderShort(): String =
    s"(TableColumnarReader $path${intervals.fold("")(i => s" (${i.intervals.length} intervals)")})"

  override def hashCode(): Int = (path, intervals).##

  override def equals(that: Any): Boolean = that match {
    case that: TableColumnarReader => path == that.path && intervals == that.intervals
    case _ => false
  }

  override def toString: String = s"TableColumnarReader($path)"

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    dataset.globals(requestedGlobalsType)

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val reader = dataset.forRead(dataset.rowsReader, intervals)
    dataset.lower(
      ctx,
      lowerGlobals(ctx, requestedType.globalType),
      intervals,
      requestedType.rowType,
      reader.contextType,
      r => dataset.readerContext(r.partition.rowFile, r),
      GetField(_, "intervals"),
    )(ReadPartition(_, _, reader))
  }
}
