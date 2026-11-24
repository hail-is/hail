package is.hail.io.columnar

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.FS
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.immutable.ArraySeq

import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

import org.apache.spark.sql.Row
import org.json4s.{JArray, JBool, JInt, JNull, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods

/** Writes tables and matrix tables in the columnar format
  * (`dev-docs/hail-query/columnar-format-spec.md`), for [[TableColumnarWriter]] and
  * [[MatrixColumnarWriter]]. Also holds the format's constants, such as its file names.
  */
object RelationalColumnarWriter {
  val FormatName: String = "hail-columnar"
  val FormatVersion: String = "0.1"
  val Container: String = "parquet"
  val ManifestFile: String = "manifest.json"
  val PartitionsFile: String = "partitions.parquet"
  val GlobalsFile: String = "globals.parquet"
  val ColsFile: String = "cols.parquet"
  val RowsDirectory: String = "rows"
  val EntriesDirectory: String = "entries"
  val DefaultRowIdField: String = "__row_id"
  val DefaultColIdxField: String = "__col_idx"
  val DefaultEntryValidField: String = "__entry_valid"

  /** A field of the partitions table: its type, and its value given one row file's write result
    * (see [[ColumnarPartitionWriter]]) and its partition index.
    */
  private case class PartitionsField(name: String, typ: PType, value: (Atom, Atom) => IR)

  /** The `start` and `end` fields of a keyed table's partitions table. */
  private def boundFields(rowPType: PStruct, key: IndexedSeq[String])
    : IndexedSeq[PartitionsField] = {
    val boundPType = rowPType.selectFields(key).setRequired(false)
    def bound(name: String, resultField: String) =
      PartitionsField(
        name,
        boundPType,
        (part, _) =>
          If(
            GetField(part, "rowCount") > I64(0),
            GetField(part, resultField),
            NA(boundPType.virtualType),
          ),
      )
    FastSeq(bound("start", "firstKey"), bound("end", "lastKey"))
  }

  private val fileRefPType: PStruct = PCanonicalStruct(
    true,
    "path" -> PCanonicalString(true),
    "container" -> PCanonicalString(true),
  )

  private def fileRef(path: IR): IR =
    MakeStruct(FastSeq("path" -> path, "container" -> Str(Container)))

  /** What a matrix table adds to a table: its columns, which the stage holds in the global field
    * `colsField`, and its entries, which each row holds in the field `entriesField`. `leafCap` is
    * the most values a leaf field of an entry file may hold in one row group, and `byteCap` the
    * most bytes a string or binary leaf field may hold in one.
    */
  case class MatrixParts(
    colsField: String,
    colKey: IndexedSeq[String],
    colPType: PStruct,
    entriesField: String,
    entryPType: PStruct,
    leafCap: Long,
    byteCap: Long,
  )

  /** Writes `stage` in the columnar format, as a table keyed by `key`, or as a matrix table given
    * `matrix`. `rowPType` and `globalPType` are the row and global types, without a matrix table's
    * entries and columns.
    *
    * Rows with equal keys never span row partitions in this format, so a stage whose partitioner
    * isn't strict for the key is strictified first. `manifest.json` is written last and is the
    * commit point.
    */
  def lower(
    ctx: ExecuteContext,
    stage: TableStage,
    key: IndexedSeq[String],
    rowPType: PStruct,
    globalPType: PStruct,
    path: String,
    overwrite: Boolean,
    matrix: Option[MatrixParts],
  ): IR = {
    assert(stage.key.startsWith(key), s"table key $key is not a prefix of ${stage.key}")
    ColumnarSchema.checkSupported("row", rowPType.virtualType)
    ColumnarSchema.checkSupported("global", globalPType.virtualType)
    matrix.foreach { m =>
      ColumnarSchema.checkSupported("column", m.colPType.virtualType)
      ColumnarSchema.checkSupported("entry", m.entryPType.virtualType)
    }
    // The reader adds a row UID field under this name, so a table with such a field can't be read.
    if (rowPType.hasField(TableReader.uidFieldName))
      fatal(
        s"columnar writer: row field '${TableReader.uidFieldName}' can't be stored, " +
          "as table readers reserve that name for row UIDs"
      )
    // Likewise, matrix table readers add a column UID field under this name.
    if (matrix.exists(_.colPType.hasField(MatrixReader.colUIDFieldName)))
      fatal(
        s"columnar writer: column field '${MatrixReader.colUIDFieldName}' can't be stored, " +
          "as matrix table readers reserve that name for column UIDs"
      )

    // Only a keyed table needs a strict partitioner and has partition bounds.
    val (ts, bounds) =
      if (key.isEmpty) (stage, FastSeq())
      else (stage.strictify(ctx, key.length - 1), boundFields(rowPType, key))
    if (ts.numPartitions > ColumnarPartitionWriter.MaxPartitions)
      fatal(
        s"columnar writer: at most ${ColumnarPartitionWriter.MaxPartitions} row partitions are " +
          s"supported, found ${ts.numPartitions}"
      )

    // Row and entry files share the Row ID field.
    val rowIdField = ColumnarSchema.internalFieldName(
      DefaultRowIdField,
      rowPType.fieldNames ++ matrix.toSeq.flatMap(_.entryPType.fieldNames),
    )
    val entries = matrix.map { m =>
      ColumnarEntries(
        m.entriesField,
        m.entryPType,
        ColumnarSchema.internalFieldName(
          DefaultEntryValidField,
          m.entryPType.fieldNames :+ rowIdField,
        ),
        m.leafCap,
        m.byteCap,
      )
    }
    val rowWriter = ColumnarPartitionWriter(path, rowPType, key, Some(rowIdField), entries)

    val partitionsFields =
      (PartitionsField("partition_index", PInt32(true), (_, idx) => idx) +: bounds) ++ FastSeq(
        PartitionsField("row_count", PInt64(true), (part, _) => GetField(part, "rowCount")),
        PartitionsField("row_file", fileRefPType, (part, _) => fileRef(GetField(part, "path"))),
      ) ++ matrix.map { _ =>
        // One entry file per column block, and there is one column block.
        PartitionsField(
          "entry_files",
          PCanonicalArray(fileRefPType, true),
          (part, _) => MakeArray(fileRef(GetField(part, "entryPath"))),
        )
      }
    val partitionsWriter = ColumnarPartitionWriter(
      path,
      PCanonicalStruct(true, partitionsFields.map(f => f.name -> f.typ): _*),
      FastSeq(),
      None,
    )

    def fileContext(relPath: IR, partitionIndex: IR): IR =
      MakeStruct(FastSeq("path" -> relPath, "partitionIndex" -> partitionIndex))

    def partitionsTable(parts: IR): IR =
      mapIR(zipWithIndex(ToStream(parts))) { elt =>
        bindIRs(GetField(elt, "elt"), GetField(elt, "idx")) { case Seq(part, idx) =>
          MakeStruct(partitionsFields.map(f => f.name -> f.value(part, idx)))
        }
      }

    val globalsWriter =
      if (globalPType.size == 0) None
      else Some(ColumnarPartitionWriter(path, globalPType, FastSeq(), None))

    // The columns file holds each column's index, then its fields (spec section 4.4).
    val cols = matrix.map { m =>
      val colIdxField = ColumnarSchema.internalFieldName(DefaultColIdxField, m.colPType.fieldNames)
      val writer = ColumnarPartitionWriter(
        path,
        PCanonicalStruct(
          true,
          (colIdxField -> PInt32(true)) +: m.colPType.fields.map(f => f.name -> f.typ): _*
        ),
        FastSeq(),
        None,
      )
      def write(globals: IR): IR =
        WritePartition(
          mapIR(zipWithIndex(ToStream(GetField(globals, m.colsField)))) { elt =>
            bindIR(GetField(elt, "elt")) { col =>
              MakeStruct(
                (colIdxField -> GetField(elt, "idx")) +:
                  m.colPType.fieldNames.map(n => n -> GetField(col, n))
              )
            }
          },
          fileContext(Str(ColsFile), I32(0)),
          writer,
        )
      (colIdxField, writer, write _)
    }

    def columnCount(globals: IR): Option[IR] =
      matrix.map(m => ArrayLen(GetField(globals, m.colsField)))

    val d = digitsNeeded(ts.numPartitions)
    val writeContexts = Literal(
      TArray(TStruct(
        FastSeq("index" -> TInt32, "prefix" -> TString) ++
          matrix.map(_ => "entryPrefix" -> TString): _*
      )),
      ArraySeq.tabulate(ts.numPartitions) { i =>
        val entryPrefix = matrix.map(_ => s"$EntriesDirectory/${partFile(d, i)}-block-0-")
        Row(i +: s"$RowsDirectory/${partFile(d, i)}-" +: entryPrefix.toSeq: _*)
      },
    )

    val job = ts.mapContexts { oldCtx =>
      zip2(oldCtx, ToStream(writeContexts), ArrayZipBehavior.AssertSameLength) { (ctx, w) =>
        MakeStruct(FastSeq("oldCtx" -> ctx, "writeCtx" -> w))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals("table_columnar_writer") {
      (rows, ctxRef) =>
        bindIRs(GetField(ctxRef, "writeCtx"), UUID4()) { case Seq(w, uuid) =>
          // The row file and its entry files share a UUID.
          def file(prefixField: String): IR = GetField(w, prefixField) + uuid + Str(".parquet")
          WritePartition(
            rows,
            MakeStruct(
              FastSeq("path" -> file("prefix"), "partitionIndex" -> GetField(w, "index")) ++
                columnCount(ts.globals).toSeq.flatMap { n =>
                  FastSeq("entryPath" -> file("entryPrefix"), "columnCount" -> n)
                }
            ),
            rowWriter,
          )
        }
    } { (parts, globals) =>
      // An unkeyed table is never distinctly keyed, as in the native format.
      val distinctlyKeyed = foldIR(ToStream(parts), Literal.coerce(TBoolean, key.nonEmpty)) {
        (acc, part) => acc && GetField(part, "distinct")
      }
      MakeStruct(
        FastSeq(
          "partitions" -> WritePartition(
            partitionsTable(parts),
            fileContext(Str(PartitionsFile), I32(0)),
            partitionsWriter,
          ),
          "distinctlyKeyed" -> distinctlyKeyed,
          "columnCount" -> columnCount(globals).getOrElse(I32(0)),
        ) ++ globalsWriter.map { w =>
          "globals" -> WritePartition(
            MakeStream(FastSeq[IR](globals), TStream(globals.typ)),
            fileContext(Str(GlobalsFile), I32(0)),
            w,
          )
        } ++ cols.map { case (_, _, write) => "cols" -> write(globals) }
      )
    }

    val manifest = ColumnarManifestWriter(
      rowsSchema = rowWriter.arrowSchema,
      key = key,
      globalsSchema = ColumnarSchema.serialize(ColumnarSchema.fileSchema(globalPType, None)),
      hasGlobalsFile = globalsWriter.isDefined,
      rowIdField = rowIdField,
      referenceGenomes = ColumnarSchema.nonBuiltinReferenceGenomes(
        FastSeq(rowPType, globalPType).map(_.virtualType) ++
          matrix.toSeq.flatMap(m => FastSeq(m.colPType, m.entryPType).map(_.virtualType)): _*
      ),
      matrix = matrix.zip(cols).zip(entries).map { case ((m, (colIdxField, colsWriter, _)), e) =>
        ColumnarMatrixManifest(
          colsSchema = colsWriter.arrowSchema,
          colKey = m.colKey,
          entriesSchema = rowWriter.entriesArrowSchema.get,
          colIdxField = colIdxField,
          entryValidField = e.entryValidField,
        )
      },
    )
    WriteMetadata(job, ColumnarCommit(path, overwrite, manifest, job.typ))
  }
}

/** What a matrix table's `manifest.json` adds to a table's, apart from the column count. */
case class ColumnarMatrixManifest(
  colsSchema: String,
  colKey: IndexedSeq[String],
  entriesSchema: String,
  colIdxField: String,
  entryValidField: String,
)

/** Writes the `manifest.json` of a table, or of a matrix table given `matrix`, once the write is
  * done.
  */
case class ColumnarManifestWriter(
  rowsSchema: String,
  key: IndexedSeq[String],
  globalsSchema: String,
  hasGlobalsFile: Boolean,
  rowIdField: String,
  referenceGenomes: IndexedSeq[ReferenceGenome],
  matrix: Option[ColumnarMatrixManifest],
) {
  import RelationalColumnarWriter._

  private def fileRef(path: String): JValue =
    JObject("path" -> JString(path), "container" -> JString(Container))

  private def strings(xs: IndexedSeq[String]): JValue = JArray(xs.map(JString(_)).toList)

  /** The manifest, given what is only known after the write. `columnCount` is ignored for a table.
    */
  def toJson(distinctlyKeyed: Boolean, columnCount: Int): JValue = {
    val matrixFields = matrix.toList.flatMap { m =>
      List(
        "cols" -> JObject(
          "schema" -> JString(m.colsSchema),
          "key" -> strings(m.colKey),
          "count" -> JInt(columnCount),
          "file" -> fileRef(ColsFile),
        ),
        "entries" -> JObject(
          "schema" -> JString(m.entriesSchema),
          "layout" -> JString("dense"),
          "column_blocks" -> JArray(List(JObject("start" -> JInt(0), "end" -> JInt(columnCount)))),
        ),
      )
    }
    val internalFields = ("row_id" -> JString(rowIdField)) :: matrix.toList.flatMap { m =>
      List("col_idx" -> JString(m.colIdxField), "entry_valid" -> JString(m.entryValidField))
    }
    JObject(
      List(
        "format" -> JString(FormatName),
        "version" -> JString(FormatVersion),
        "kind" -> JString(if (matrix.isDefined) "matrix_table" else "table"),
        "rows" -> JObject(
          "schema" -> JString(rowsSchema),
          "key" -> strings(key),
          "distinctly_keyed" -> JBool(distinctlyKeyed),
        ),
        "globals" -> JObject(
          "schema" -> JString(globalsSchema),
          "file" -> (if (hasGlobalsFile) fileRef(GlobalsFile) else JNull),
        ),
        "partitions" -> fileRef(PartitionsFile),
        "internal_fields" -> JObject(internalFields),
        "reference_genomes" -> JArray(referenceGenomes.map(_.toJSON).toList),
      ) ++ matrixFields
    )
  }

  def write(fs: FS, path: String, distinctlyKeyed: Boolean, columnCount: Int): Unit =
    using(new OutputStreamWriter(fs.create(s"$path/$ManifestFile"), StandardCharsets.UTF_8)) {
      out => out.write(JsonMethods.compact(toJson(distinctlyKeyed, columnCount)))
    }
}

/** Prepares the output directory, runs the write, and then commits it by writing `manifest.json`.
  * The annotation is the write's result, a struct with a boolean `distinctlyKeyed` field and an
  * int32 `columnCount` field.
  */
case class ColumnarCommit(
  path: String,
  overwrite: Boolean,
  manifest: ColumnarManifestWriter,
  annotationType: Type,
) extends MetadataWriter {
  import RelationalColumnarWriter._

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    val fs = cb.emb.getFS
    if (overwrite) {
      // Uncommit any existing dataset first, so a delete that fails partway never leaves a
      // manifest whose files are gone (spec section 2.1).
      cb += fs.invoke[String, Boolean, Unit]("delete", s"$path/$ManifestFile", false)
      cb += fs.invoke[String, Boolean, Unit]("delete", path, true)
    } else
      cb.if_(
        fs.invoke[String, Boolean]("exists", path),
        cb._fatal(s"columnar writer: file already exists: $path"),
      )
    cb += fs.invoke[String, Unit]("mkDir", path)
    cb += fs.invoke[String, Unit]("mkDir", s"$path/$RowsDirectory")
    if (manifest.matrix.isDefined)
      cb += fs.invoke[String, Unit]("mkDir", s"$path/$EntriesDirectory")

    val result = writeAnnotations.getOrAssert(cb).asBaseStruct
    def field(name: String): SValue = result.loadField(cb, name).getOrAssert(cb)
    cb += cb.emb.getObject(manifest).invoke[FS, String, Boolean, Int, Unit](
      "write",
      fs,
      path,
      field("distinctlyKeyed").asBoolean.value,
      field("columnCount").asInt.value,
    )
  }
}
