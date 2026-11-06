package is.hail.io.columnar

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.TableStage
import is.hail.io.fs.FS
import is.hail.types.RTable
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.immutable.ArraySeq

import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

import org.apache.spark.sql.Row
import org.json4s.{JArray, JBool, JNull, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods

object TableColumnarWriter {
  val FormatName: String = "hail-columnar"
  val FormatVersion: String = "0.1"
  val Container: String = "parquet"
  val ManifestFile: String = "manifest.json"
  val PartitionsFile: String = "partitions.parquet"
  val GlobalsFile: String = "globals.parquet"
  val RowsDirectory: String = "rows"
  val DefaultRowIdField: String = "__row_id"

  /** Row IDs hold the row partition index above bit 40. */
  val MaxPartitions: Int = 1 << 23
}

/** Writes a table in the columnar format (`dev-docs/hail-query/columnar-format-spec.md`).
  *
  * Rows with equal keys never span row partitions in this format, so a stage whose partitioner
  * isn't strict for the table key is strictified first. `manifest.json` is written last and is the
  * commit point.
  */
case class TableColumnarWriter(path: String, overwrite: Boolean = false) extends TableWriter {
  import TableColumnarWriter._

  // The stage's partitioner key may extend the table key, and an RTable built from the stage
  // carries the partitioner key, so pass the table key explicitly.
  override def apply(ctx: ExecuteContext, tv: TableValue): Unit = {
    val stage = TableValueIntermediate(tv).asTableStage(ctx)
    CompileAndEvaluate(ctx, lower(ctx, stage, RTable.fromTableStage(ctx, stage), tv.typ.key))
  }

  // When lowering a TableWrite, `r` is the requiredness of the child TableIR, whose key is the
  // table key.
  override def lower(ctx: ExecuteContext, stage: TableStage, r: RTable): IR =
    lower(ctx, stage, r, r.key.toIndexedSeq)

  private def lower(ctx: ExecuteContext, stage: TableStage, r: RTable, key: IndexedSeq[String])
    : IR = {
    assert(stage.key.startsWith(key), s"table key $key is not a prefix of ${stage.key}")
    ColumnarSchema.checkSupported("row", stage.rowType)
    ColumnarSchema.checkSupported("global", stage.globalType)

    val ts = if (key.isEmpty) stage else stage.strictify(ctx, key.length - 1)
    if (ts.numPartitions > MaxPartitions)
      fatal(s"columnar writer: at most $MaxPartitions row partitions are supported, " +
        s"found ${ts.numPartitions}")

    val rowPType = r.rowType.canonicalPType(ts.rowType).asInstanceOf[PStruct]
    val globalPType = r.globalType.canonicalPType(ts.globalType).asInstanceOf[PStruct]
    val rowIdField = ColumnarSchema.internalFieldName(DefaultRowIdField, ts.rowType.fieldNames)

    val rowWriter = ColumnarPartitionWriter(path, rowPType, key, Some(rowIdField))

    val bounds: Seq[(String, PType)] =
      if (key.isEmpty) FastSeq()
      else {
        val keyPType = rowPType.selectFields(key).setRequired(false)
        FastSeq("start" -> keyPType, "end" -> keyPType)
      }
    val partitionsPType = PCanonicalStruct(
      true,
      (("partition_index" -> PInt32(true)) +: bounds) ++ FastSeq(
        "row_count" -> PInt64(true),
        "row_file" -> PCanonicalStruct(
          true,
          "path" -> PCanonicalString(true),
          "container" -> PCanonicalString(true),
        ),
      ): _*
    )
    val partitionsWriter = ColumnarPartitionWriter(path, partitionsPType, FastSeq(), None)

    val globalsWriter =
      if (globalPType.size == 0) None
      else Some(ColumnarPartitionWriter(path, globalPType, FastSeq(), None))

    def fileContext(relPath: IR, partitionIndex: IR): IR =
      MakeStruct(FastSeq("path" -> relPath, "partitionIndex" -> partitionIndex))

    val keyType = ts.rowType.typeAfterSelectNames(key)

    def partitionsTable(parts: IR): IR =
      mapIR(zipWithIndex(ToStream(parts))) { elt =>
        bindIRs(GetField(elt, "elt"), GetField(elt, "idx")) { case Seq(part, idx) =>
          def nonEmpty: IR = GetField(part, "rowCount") > I64(0)
          def bound(field: String): IR = If(nonEmpty, GetField(part, field), NA(keyType))
          val boundFields =
            if (key.isEmpty) FastSeq()
            else FastSeq("start" -> bound("firstKey"), "end" -> bound("lastKey"))
          MakeStruct(
            (("partition_index" -> (idx: IR)) +: boundFields) ++ FastSeq(
              "row_count" -> GetField(part, "rowCount"),
              "row_file" -> MakeStruct(FastSeq(
                "path" -> GetField(part, "path"),
                "container" -> Str(Container),
              )),
            )
          )
        }
      }

    val d = digitsNeeded(ts.numPartitions)
    val writeContexts = Literal(
      TArray(TStruct("prefix" -> TString, "index" -> TInt32)),
      ArraySeq.tabulate(ts.numPartitions)(i => Row(s"$RowsDirectory/${partFile(d, i)}-", i)),
    )

    val job = ts.mapContexts { oldCtx =>
      zip2(oldCtx, ToStream(writeContexts), ArrayZipBehavior.AssertSameLength) { (ctx, w) =>
        MakeStruct(FastSeq("oldCtx" -> ctx, "writeCtx" -> w))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals("table_columnar_writer") {
      (rows, ctxRef) =>
        bindIR(GetField(ctxRef, "writeCtx")) { w =>
          WritePartition(
            rows,
            fileContext(GetField(w, "prefix") + UUID4() + Str(".parquet"), GetField(w, "index")),
            rowWriter,
          )
        }
    } { (parts, globals) =>
      val distinctlyKeyed =
        if (key.isEmpty) False()
        else foldIR(ToStream(parts), True())((acc, part) => acc && GetField(part, "distinct"))
      MakeStruct(
        FastSeq(
          "partitions" -> WritePartition(
            partitionsTable(parts),
            fileContext(Str(PartitionsFile), I32(0)),
            partitionsWriter,
          ),
          "distinctlyKeyed" -> distinctlyKeyed,
        ) ++ globalsWriter.map { w =>
          "globals" -> WritePartition(
            MakeStream(FastSeq[IR](globals), TStream(globals.typ)),
            fileContext(Str(GlobalsFile), I32(0)),
            w,
          )
        }
      )
    }

    val manifest = ColumnarTableManifest(
      rowsSchema = rowWriter.arrowSchema,
      key = key,
      globalsSchema = ColumnarSchema.serialize(ColumnarSchema.fileSchema(globalPType, None)),
      hasGlobalsFile = globalsWriter.isDefined,
      rowIdField = rowIdField,
    )
    WriteMetadata(job, ColumnarTableCommit(path, overwrite, manifest, job.typ))
  }
}

/** The contents of a table's `manifest.json`, apart from what is only known after the write. */
case class ColumnarTableManifest(
  rowsSchema: String,
  key: IndexedSeq[String],
  globalsSchema: String,
  hasGlobalsFile: Boolean,
  rowIdField: String,
) {
  import TableColumnarWriter._

  private def fileRef(path: String): JValue =
    JObject("path" -> JString(path), "container" -> JString(Container))

  def toJson(distinctlyKeyed: Boolean): JValue =
    JObject(
      "format" -> JString(FormatName),
      "version" -> JString(FormatVersion),
      "kind" -> JString("table"),
      "rows" -> JObject(
        "schema" -> JString(rowsSchema),
        "key" -> JArray(key.map(JString(_)).toList),
        "distinctly_keyed" -> JBool(distinctlyKeyed),
      ),
      "globals" -> JObject(
        "schema" -> JString(globalsSchema),
        "file" -> (if (hasGlobalsFile) fileRef(GlobalsFile) else JNull),
      ),
      "partitions" -> fileRef(PartitionsFile),
      "internal_fields" -> JObject("row_id" -> JString(rowIdField)),
      "reference_genomes" -> JArray(Nil),
    )

  def write(fs: FS, path: String, distinctlyKeyed: Boolean): Unit =
    using(new OutputStreamWriter(fs.create(s"$path/$ManifestFile"), StandardCharsets.UTF_8)) {
      out => out.write(JsonMethods.compact(toJson(distinctlyKeyed)))
    }
}

/** Prepares the output directory, runs the write, and then commits it by writing `manifest.json`.
  * The annotation is the write's result, a struct with a boolean `distinctlyKeyed` field.
  */
case class ColumnarTableCommit(
  path: String,
  overwrite: Boolean,
  manifest: ColumnarTableManifest,
  annotationType: Type,
) extends MetadataWriter {

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    val fs = cb.emb.getFS
    if (overwrite)
      cb += fs.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.if_(
        fs.invoke[String, Boolean]("exists", path),
        cb._fatal(s"columnar writer: file already exists: $path"),
      )
    cb += fs.invoke[String, Unit]("mkDir", path)
    cb += fs.invoke[String, Unit]("mkDir", s"$path/${TableColumnarWriter.RowsDirectory}")

    val result = writeAnnotations.getOrAssert(cb).asBaseStruct
    val distinctlyKeyed = result.loadField(cb, "distinctlyKeyed").getOrAssert(cb).asBoolean.value
    cb += cb.emb.getObject(manifest)
      .invoke[FS, String, Boolean, Unit]("write", fs, path, distinctlyKeyed)
  }
}
