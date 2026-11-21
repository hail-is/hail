package is.hail.io.columnar

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{
  Atom, Literal, MakeStruct, ReadPartition, SelectFields, Str, ToArray, ToStream,
}
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.rvd.RVDPartitioner
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{fatal, Interval}
import is.hail.variant.ReferenceGenomeEnv

import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters._

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.spark.sql.Row

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

  /** Fails unless `partitions` has exactly the fields of the partitions table of a table, or of a
    * matrix table if `matrix`, whose row schema is `rows` and whose key is `key`, with their types.
    * The bounds must have exactly the Arrow types of the key fields, apart from field IDs.
    */
  def checkSchema(
    partitions: Schema,
    rows: Schema,
    key: IndexedSeq[String],
    matrix: Boolean = false,
  ): Unit = {
    def reject(msg: String): Nothing = fatal(s"columnar reader: partitions table: $msg")

    val keyFields = key.map(k => withoutFieldIds(rows.findField(k)))
    def isBound(f: Field): Boolean =
      f.isNullable && f.getType == ArrowType.Struct.INSTANCE &&
        f.getChildren.asScala.map(withoutFieldIds) == keyFields
    def isFileRef(f: Field): Boolean =
      f.getType == ArrowType.Struct.INSTANCE &&
        f.getChildren.asScala.map(_.getName) == Seq("path", "container") &&
        f.getChildren.asScala.forall(isUtf8)

    val expected: Seq[(String, Field => Boolean)] =
      Seq[(String, Field => Boolean)](
        "partition_index" -> (f => !f.isNullable && f.getType == new ArrowType.Int(32, true)),
        "row_count" -> (f => !f.isNullable && f.getType == new ArrowType.Int(64, true)),
        "row_file" -> (f => !f.isNullable && isFileRef(f)),
      ) ++ (if (key.nonEmpty) Seq("start" -> isBound _, "end" -> isBound _) else Seq()) ++
        (if (matrix)
           Seq[(String, Field => Boolean)](
             "entry_files" -> (f =>
               !f.isNullable && f.getType == ArrowType.List.INSTANCE &&
                 f.getChildren.size == 1 && isFileRef(f.getChildren.get(0))
             )
           )
         else Seq())

    val kind = (if (key.isEmpty) "an unkeyed " else "a keyed ") +
      (if (matrix) "matrix table" else "table")
    val fields = partitions.getFields.asScala.map(f => f.getName -> f).toMap
    partitions.getFields.asScala.map(_.getName).find(n => !expected.exists(_._1 == n)).foreach {
      n => reject(s"unexpected field '$n' for $kind")
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
  * table or an empty partition. `entryFiles` holds a matrix table's entry files, one per column
  * block, and is empty for a table.
  */
private case class ColumnarPartition(
  rowFile: String,
  rowCount: Long,
  start: Row,
  end: Row,
  entryFiles: IndexedSeq[String],
) {

  /** The keys of a keyed table's non-empty row partition. */
  def bounds: Interval = Interval(start, end, includesStart = true, includesEnd = true)
}

/** A row partition to read, and in a key-range read, the parts of the key intervals within its
  * bounds: sorted, disjoint and never empty.
  */
private case class ColumnarPartitionRead(
  partition: ColumnarPartition,
  intervals: Option[IndexedSeq[Interval]],
)

/** How a read partitions its rows: its partitioner, and for each of its partitions, the row
  * partition reads whose rows it holds, in order.
  */
private case class ColumnarPartitioning(
  partitioner: RVDPartitioner,
  reads: IndexedSeq[IndexedSeq[ColumnarPartitionRead]],
)

/** What the table and matrix table readers read alike, from the columnar dataset at `path` whose
  * manifest is `manifest`: the partitions table, the row files and the globals file. Row files are
  * read with the Row ID as the UID `rowUIDField`.
  */
private class ColumnarDataset(
  val path: String,
  val manifest: ColumnarManifest,
  val references: ReferenceGenomeEnv,
  rowUIDField: String,
) {
  val rowFilePType: PStruct = ArrowSchemaConversion.fromArrowSchema(manifest.rowsSchema, references)

  val globalsPType: PStruct =
    ArrowSchemaConversion.fromArrowSchema(manifest.globalsSchema, references)

  val rowsReader: ColumnarPartitionReader = ColumnarPartitionReader(
    rowFilePType,
    ColumnarSchema.serialize(manifest.rowsSchema),
    Some(manifest.rowIdField),
    rowUIDField,
  )

  /** The row fields, without the Row ID. */
  def rowType: TStruct = rowFilePType.virtualType.deleteKey(manifest.rowIdField)

  def absolute(relPath: String): String = s"$path/$relPath"

  /** A reader of a file whose fields are `pt`, with Arrow schema `schema`, and which has no Row ID
    * field. Its UID is the row's position, under a name that is none of its fields'.
    */
  def wholeFileReader(schema: Schema, pt: PStruct): ColumnarPartitionReader =
    ColumnarPartitionReader(
      pt,
      ColumnarSchema.serialize(schema),
      None,
      ColumnarSchema.internalFieldName(rowUIDField, pt.fieldNames),
    )

  /** The rows of type `requestedType` of the file at `relPath`, read with `reader`. */
  def readWholeFile(relPath: String, reader: ColumnarPartitionReader, requestedType: TStruct): IR =
    ToArray(ReadPartition(
      MakeStruct(FastSeq("path" -> Str(absolute(relPath)))),
      requestedType,
      reader,
    ))

  /** The globals of type `requestedType`. */
  def globals(requestedType: TStruct): IR =
    manifest.globalsFile match {
      case None => MakeStruct(FastSeq())
      case Some(file) =>
        readWholeFile(
          file,
          wholeFileReader(manifest.globalsSchema, globalsPType),
          requestedType,
        ).at(0)
    }

  def globalsRequiredness(requestedType: TStruct): PType =
    ColumnarReadPlan.subset(globalsPType, requestedType)

  /** The row partitions that have rows, from the partitions table, which is read on the driver. Its
    * schema isn't in the manifest, so it is taken from the file's footer.
    */
  def nonEmptyPartitions(ctx: ExecuteContext): IndexedSeq[ColumnarPartition] = {
    val file = manifest.partitionsFile
    val schema = ColumnarFileSource.readArrowSchema(ctx.fs, absolute(file))
    val matrix = manifest.matrix.isDefined
    ColumnarPartitionsTable.checkSchema(schema, manifest.rowsSchema, manifest.key, matrix)
    val pt = ArrowSchemaConversion.fromArrowSchema(schema, references)
    val t = pt.virtualType
    val rows = CompileAndEvaluate[IndexedSeq[Row]](
      ctx,
      readWholeFile(file, wholeFileReader(schema, pt), t),
    )
    val keyed = manifest.key.nonEmpty
    val parts = rows.zipWithIndex.map { case (r, i) =>
      def field[T](name: String): T = r.getAs[T](t.fieldIdx(name))
      if (field[Int]("partition_index") != i)
        fatal(s"columnar reader: partitions table row $i has partition_index " +
          s"${field[Int]("partition_index")}")
      def fileRef(r: Row, what: String): String = r match {
        case Row(path: String, RelationalColumnarWriter.Container) => path
        case Row(_: String, container: String) =>
          fatal(s"columnar reader: $what of partition $i has unsupported container '$container'")
        case _ => fatal(s"columnar reader: $what of partition $i is missing")
      }
      val entryFiles =
        if (!matrix) FastSeq()
        else {
          val files = field[IndexedSeq[Row]]("entry_files")
          // A 0.1 matrix table has one column block (see ColumnarManifest).
          if (files.length != 1)
            fatal(s"columnar reader: partition $i has ${files.length} entry files, but the " +
              "matrix table has 1 column block")
          files.map(fileRef(_, "an entry file"))
        }
      ColumnarPartition(
        fileRef(field[Row]("row_file"), "the row file"),
        field[Long]("row_count"),
        if (keyed) field[Row]("start") else null,
        if (keyed) field[Row]("end") else null,
        entryFiles,
      )
    }
    parts.filter(_.rowCount > 0)
  }

  def keyType: TStruct = rowType.typeAfterSelectNames(manifest.key)

  /** How a read of `intervals` partitions the rows of the row partitions that have rows:
    *   - with no intervals, each row partition is a partition;
    *   - with `filterIntervals`, each row partition that overlaps the intervals is a partition,
    *     narrowed to them;
    *   - otherwise each interval is a partition, of its rows from every row partition it overlaps.
    */
  def partitioning(ctx: ExecuteContext, intervals: Option[ColumnarReadIntervals])
    : ColumnarPartitioning = {
    val parts = nonEmptyPartitions(ctx)
    if (intervals.isDefined && manifest.key.isEmpty)
      fatal(s"columnar reader '$path': a key-range read of an unkeyed table")
    intervals match {
      case None =>
        ColumnarPartitioning(
          if (manifest.key.isEmpty) RVDPartitioner.unkeyed(parts.length)
          else new RVDPartitioner(keyType, parts.map(_.bounds)),
          parts.map(p => FastSeq(ColumnarPartitionRead(p, None))),
        )
      case Some(ColumnarReadIntervals(intervals, true)) =>
        val query = new RVDPartitioner(keyType, intervals)
        val reads = parts.flatMap { p =>
          val within = query.queryInterval(p.bounds)
            .flatMap(i => intervals(i).intersect(query.kord, p.bounds))
          if (within.isEmpty) None else Some(ColumnarPartitionRead(p, Some(within)))
        }
        ColumnarPartitioning(
          new RVDPartitioner(
            keyType,
            reads.map { r =>
              val within = r.intervals.get
              Interval(within.head.left, within.last.right)
            },
          ),
          reads.map(FastSeq(_)),
        )
      case Some(ColumnarReadIntervals(intervals, false)) =>
        val stored = new RVDPartitioner(keyType, parts.map(_.bounds))
        ColumnarPartitioning(
          new RVDPartitioner(keyType, intervals),
          intervals.map { i =>
            stored.queryInterval(i).flatMap { j =>
              val p = parts(j)
              i.intersect(stored.kord, p.bounds).map(w =>
                ColumnarPartitionRead(p, Some(FastSeq(w)))
              )
            }
          },
        )
    }
  }

  /** `reader`, a reader of this dataset's row or entry files, for a read of `intervals`: in a
    * key-range read, one that reads only the row groups and pages that may hold keys in them.
    */
  def forRead(reader: ColumnarPartitionReader, intervals: Option[ColumnarReadIntervals])
    : ColumnarPartitionReader =
    if (intervals.isEmpty) reader else reader.copy(keyRange = Some(ColumnarKeyRange(keyType)))

  /** The context of a reader from [[forRead]] of `file`, a row or entry file of `read`'s row
    * partition.
    */
  def readerContext(file: String, read: ColumnarPartitionRead): Row =
    read.intervals match {
      case None => Row(absolute(file))
      case Some(intervals) =>
        Row(
          absolute(file),
          absolute(read.partition.rowFile),
          intervals.map(RVDPartitioner.intervalToIRRepresentation(_, keyType.size)),
        )
    }

  /** The stage that reads rows of type `rowType` with key-range read `intervals`, if any.
    *
    * Each row partition read has a context of type `contextType`, given by `context`, from which
    * `read` reads rows of a given type, which includes the key in a key-range read. In a key-range
    * read, `keyIntervals` gets the key intervals of a context from [[readerContext]], and the rows
    * read are filtered to them.
    */
  def lower(
    ctx: ExecuteContext,
    globals: IR,
    intervals: Option[ColumnarReadIntervals],
    rowType: TStruct,
    contextType: TStruct,
    context: ColumnarPartitionRead => Row,
    keyIntervals: Atom => IR,
  )(
    read: (Atom, TStruct) => IR
  ): TableStage = {
    val ColumnarPartitioning(partitioner, reads) = partitioning(ctx, intervals)
    TableStage(
      globals,
      partitioner,
      TableStageDependency.none,
      ToStream(Literal(
        TArray(TArray(contextType)),
        ArraySeq.from(reads.map(rs => ArraySeq.from(rs.map(context)))),
      )),
      contexts =>
        flatMapIR(ToStream(contexts, true)) { c =>
          filterToKeyIntervals(rowType, intervals.map(_ => keyIntervals(c)))(read(c, _))
        },
    )
  }

  /** The rows of type `rowType` that `read` gives, filtered in a key-range read to those whose key
    * is in `intervals`. `read` takes the row type to read, which then includes the key.
    */
  private def filterToKeyIntervals(rowType: TStruct, intervals: Option[IR])(read: TStruct => IR)
    : IR =
    intervals match {
      case None => read(rowType)
      case Some(intervals) =>
        val missingKey = manifest.key.filterNot(rowType.hasField)
        val readType = TStruct(
          rowType.fields.map(f => f.name -> f.typ) ++
            missingKey.map(k => k -> this.rowType.fieldType(k)): _*
        )
        val rows = filterIR(read(readType)) { row =>
          invoke("partitionerContains", TBoolean, intervals, SelectFields(row, manifest.key))
        }
        if (missingKey.isEmpty) rows
        else mapIR(rows)(row => SelectFields(row, rowType.fieldNames))
    }
}
