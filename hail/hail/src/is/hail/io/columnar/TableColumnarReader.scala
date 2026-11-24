package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{Literal, MakeStruct, ReadPartition, Str, ToArray, ToStream}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.io.fs.FS
import is.hail.rvd.RVDPartitioner
import is.hail.types.VirtualTypeWithReq
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{fatal, Interval}
import is.hail.variant.ReferenceGenomeEnv

import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters._

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Formats, JObject, JString, JValue}

/** Rules for the partitions table (spec section 4.1). */
object ColumnarPartitionsTable {

  /** `f` without field IDs, at any depth. */
  private def withoutFieldIds(f: Field): Field = {
    val metadata = new java.util.LinkedHashMap[String, String]()
    if (f.getMetadata != null) metadata.putAll(f.getMetadata)
    metadata.remove(ColumnarSchema.FieldIdKey)
    new Field(
      f.getName,
      new FieldType(f.isNullable, f.getType, f.getDictionary, metadata),
      f.getChildren.asScala.map(withoutFieldIds).asJava,
    )
  }

  private def isUtf8(f: Field): Boolean =
    f.getType == ArrowType.Utf8.INSTANCE && f.getChildren.isEmpty

  /** Fails unless `partitions` has exactly the fields of a table's partitions table whose row
    * schema is `rows` and whose key is `key`, with their types. The bounds must have exactly the
    * Arrow types of the key fields, apart from field IDs.
    */
  def checkSchema(partitions: Schema, rows: Schema, key: IndexedSeq[String]): Unit = {
    def reject(msg: String): Nothing = fatal(s"columnar reader: partitions table: $msg")

    val keyFields = key.map(k => withoutFieldIds(rows.findField(k)))
    def isBound(f: Field): Boolean =
      f.isNullable && f.getType == ArrowType.Struct.INSTANCE &&
        f.getChildren.asScala.map(withoutFieldIds) == keyFields

    val expected: Seq[(String, Field => Boolean)] =
      Seq[(String, Field => Boolean)](
        "partition_index" -> (f => !f.isNullable && f.getType == new ArrowType.Int(32, true)),
        "row_count" -> (f => !f.isNullable && f.getType == new ArrowType.Int(64, true)),
        "row_file" -> (f =>
          !f.isNullable && f.getType == ArrowType.Struct.INSTANCE &&
            f.getChildren.asScala.map(_.getName) == Seq("path", "container") &&
            f.getChildren.asScala.forall(isUtf8)
        ),
      ) ++ (if (key.nonEmpty) Seq("start" -> isBound _, "end" -> isBound _) else Seq())

    val fields = partitions.getFields.asScala.map(f => f.getName -> f).toMap
    partitions.getFields.asScala.map(_.getName).find(n => !expected.exists(_._1 == n)).foreach {
      n => reject(s"unexpected field '$n' for a${if (key.isEmpty) "n unkeyed" else " keyed"} table")
    }
    expected.foreach { case (name, ok) =>
      fields.get(name) match {
        case None => reject(s"missing field '$name'")
        case Some(f) if !ok(f) => reject(s"field '$name' has the wrong type: $f")
        case _ =>
      }
    }
  }
}

/** A row partition, as listed in the partitions table. `start` and `end` are missing for an unkeyed
  * table or an empty partition.
  */
private case class ColumnarPartition(rowFile: String, rowCount: Long, start: Row, end: Row)

object TableColumnarReader {

  /** A reader of the columnar table at `path`. Its reference genomes must be equal to any of the
    * same name in `references`.
    */
  def apply(fs: FS, references: ReferenceGenomeEnv, path: String): TableColumnarReader = {
    val manifest = ColumnarManifest.read(fs, path)
    val rgs = references.withReferences(manifest.referenceGenomes)
    new TableColumnarReader(
      path,
      manifest,
      ArrowSchemaConversion.fromArrowSchema(manifest.rowsSchema, rgs),
      ArrowSchemaConversion.fromArrowSchema(manifest.globalsSchema, rgs),
      rgs,
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
class TableColumnarReader private (
  val path: String,
  manifest: ColumnarManifest,
  rowFilePType: PStruct,
  globalsPType: PStruct,
  references: ReferenceGenomeEnv,
) extends TableReaderWithExtraUID {

  private val rowsReader = ColumnarPartitionReader(
    rowFilePType,
    ColumnarSchema.serialize(manifest.rowsSchema),
    Some(manifest.rowIdField),
    uidFieldName,
  )

  private def absolute(relPath: String): String = s"$path/$relPath"

  /** The rows of type `requestedType` of the file at `relPath`, which has no Row ID field, whose
    * Arrow schema is `schema` and whose Hail type is `pt`.
    */
  private def readWholeFile(relPath: String, schema: Schema, pt: PStruct, requestedType: TStruct)
    : IR = {
    // Such a file's UID is the row's position, under a name that is none of its fields'.
    val uidField = ColumnarSchema.internalFieldName(uidFieldName, pt.fieldNames)
    val reader = ColumnarPartitionReader(pt, ColumnarSchema.serialize(schema), None, uidField)
    ToArray(ReadPartition(
      MakeStruct(FastSeq("path" -> Str(absolute(relPath)))),
      requestedType,
      reader,
    ))
  }

  override def pathsUsed: Seq[String] = FastSeq(path)

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def isDistinctlyKeyed: Boolean = manifest.distinctlyKeyed

  override def uidType: Type = TInt64

  override def fullTypeWithoutUIDs: TableType = {
    val rowType = rowsReader.fullRowType.deleteKey(uidFieldName)
    TableType(rowType, manifest.key, globalsPType.virtualType)
  }

  override protected def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(rowsReader.rowPType(requestedType.rowType))

  override protected def uidRequiredness: VirtualTypeWithReq = VirtualTypeWithReq(PInt64(true))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(ColumnarReadPlan.subset(globalsPType, requestedType.globalType))

  override def toJValue: JValue =
    JObject("name" -> JString("TableColumnarReader"), "path" -> JString(path))

  override def renderShort(): String = s"(TableColumnarReader $path)"

  override def hashCode(): Int = path.hashCode

  override def equals(that: Any): Boolean = that match {
    case that: TableColumnarReader => path == that.path
    case _ => false
  }

  override def toString: String = s"TableColumnarReader($path)"

  /** The partitions table, read on the driver. Its schema isn't in the manifest, so it is taken
    * from the file's footer.
    */
  private def partitions(ctx: ExecuteContext): IndexedSeq[ColumnarPartition] = {
    val file = manifest.partitionsFile
    val schema = ColumnarFileSource.readArrowSchema(ctx.fs, absolute(file))
    ColumnarPartitionsTable.checkSchema(schema, manifest.rowsSchema, manifest.key)
    val pt = ArrowSchemaConversion.fromArrowSchema(schema, references)
    val t = pt.virtualType
    val rows = CompileAndEvaluate[IndexedSeq[Row]](ctx, readWholeFile(file, schema, pt, t))
    val keyed = manifest.key.nonEmpty
    rows.zipWithIndex.map { case (r, i) =>
      def field[T](name: String): T = r.getAs[T](t.fieldIdx(name))
      if (field[Int]("partition_index") != i)
        fatal(s"columnar reader: partitions table row $i has partition_index " +
          s"${field[Int]("partition_index")}")
      val Row(rowFile: String, container: String) = field[Row]("row_file"): @unchecked
      if (container != TableColumnarWriter.Container)
        fatal(s"columnar reader: row file of partition $i has unsupported container '$container'")
      ColumnarPartition(
        rowFile,
        field[Long]("row_count"),
        if (keyed) field[Row]("start") else null,
        if (keyed) field[Row]("end") else null,
      )
    }
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    manifest.globalsFile match {
      case None => MakeStruct(FastSeq())
      case Some(file) =>
        readWholeFile(file, manifest.globalsSchema, globalsPType, requestedGlobalsType).at(0)
    }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val parts = partitions(ctx).filter(_.rowCount > 0)
    val partitioner =
      if (manifest.key.isEmpty) RVDPartitioner.unkeyed(parts.length)
      else
        new RVDPartitioner(
          fullType.keyType,
          parts.map(p => Interval(p.start, p.end, includesStart = true, includesEnd = true)),
        )
    TableStage(
      lowerGlobals(ctx, requestedType.globalType),
      partitioner,
      TableStageDependency.none,
      ToStream(Literal(
        TArray(rowsReader.contextType),
        ArraySeq.from(parts.map(p => Row(absolute(p.rowFile)))),
      )),
      ctx => ReadPartition(ctx, requestedType.rowType, rowsReader),
    )
  }
}
