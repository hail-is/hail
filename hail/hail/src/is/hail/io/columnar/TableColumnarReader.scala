package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{Literal, ReadPartition, ToStream}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.io.fs.FS
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.fatal
import is.hail.variant.ReferenceGenomeEnv

import scala.collection.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JObject, JString, JValue}

object TableColumnarReader {

  /** A reader of the columnar table at `path`. Its reference genomes must be equal to any of the
    * same name in `references`.
    */
  def apply(fs: FS, references: ReferenceGenomeEnv, path: String): TableColumnarReader = {
    val manifest = ColumnarManifest.read(fs, path)
    if (manifest.matrix.isDefined)
      fatal(s"columnar table '$path': it is a matrix table, not a table")
    new TableColumnarReader(
      new ColumnarDataset(
        path,
        manifest,
        references.withReferences(manifest.referenceGenomes),
        TableReader.uidFieldName,
      )
    )
  }

  def fromJValue(fs: FS, references: ReferenceGenomeEnv, jv: JValue): TableColumnarReader = {
    implicit val formats: Formats = DefaultFormats
    TableColumnarReader(fs, references, (jv \ "path").extract[String])
  }
}

/** Reads a table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`).
  *
  * The key, partitioning and distinct keying come from the manifest, so the table is never
  * re-sorted. Row partitions with no rows are dropped. The row UID is the Row ID.
  */
class TableColumnarReader private (dataset: ColumnarDataset) extends TableReaderWithExtraUID {

  def path: String = dataset.path

  override def pathsUsed: Seq[String] = FastSeq(path)

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

  override def renderShort(): String = s"(TableColumnarReader $path)"

  override def hashCode(): Int = path.hashCode

  override def equals(that: Any): Boolean = that match {
    case that: TableColumnarReader => path == that.path
    case _ => false
  }

  override def toString: String = s"TableColumnarReader($path)"

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    dataset.globals(requestedGlobalsType)

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val parts = dataset.nonEmptyPartitions(ctx)
    TableStage(
      lowerGlobals(ctx, requestedType.globalType),
      dataset.partitioner(fullType.keyType, parts),
      TableStageDependency.none,
      ToStream(Literal(
        TArray(dataset.rowsReader.contextType),
        ArraySeq.from(parts.map(p => Row(dataset.absolute(p.rowFile)))),
      )),
      ctx => ReadPartition(ctx, requestedType.rowType, dataset.rowsReader),
    )
  }
}
