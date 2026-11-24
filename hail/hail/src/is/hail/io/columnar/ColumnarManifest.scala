package is.hail.io.columnar

import is.hail.io.fs.FS
import is.hail.utils.{fatal, using}
import is.hail.variant.ReferenceGenome

import scala.io.Source
import scala.jdk.CollectionConverters._

import java.nio.charset.StandardCharsets

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, Schema}
import org.json4s.{JArray, JBool, JInt, JNull, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods

/** What a matrix table's `manifest.json` adds to a table's, as a reader sees it (spec section 3).
  * Its entries are in the dense layout, in one column block that holds every column.
  */
case class ColumnarMatrixComponents(
  colsSchema: Schema,
  colKey: IndexedSeq[String],
  colCount: Int,
  colsFile: String,
  entriesSchema: Schema,
  colIdxField: String,
  entryValidField: String,
)

/** A table's or matrix table's `manifest.json`, as a reader sees it (spec section 3). `matrix` is
  * defined for a matrix table. Paths are relative to the dataset's directory.
  */
case class ColumnarManifest(
  rowsSchema: Schema,
  key: IndexedSeq[String],
  distinctlyKeyed: Boolean,
  globalsSchema: Schema,
  globalsFile: Option[String],
  partitionsFile: String,
  rowIdField: String,
  referenceGenomes: IndexedSeq[ReferenceGenome],
  matrix: Option[ColumnarMatrixComponents],
)

object ColumnarManifest {
  import RelationalColumnarWriter._

  /** Reads and validates `<path>/manifest.json`. A directory without one is not a committed table
    * or matrix table (spec section 2.1).
    */
  def read(fs: FS, path: String): ColumnarManifest = {
    val manifestPath = s"$path/$ManifestFile"
    if (!fs.isFile(manifestPath))
      fatal(s"'$path' is not a columnar table: it has no $ManifestFile")
    val json = using(Source.fromInputStream(fs.open(manifestPath), StandardCharsets.UTF_8.name))(
      _.mkString
    )
    parse(json, s"columnar dataset '$path'")
  }

  private class ManifestException(msg: String) extends RuntimeException(msg)

  private def reject(msg: String): Nothing = throw new ManifestException(msg)

  /** Parses and validates the contents of a `manifest.json`. Fails on anything a `0.x` reader must
    * reject: another format, version, kind or container, and missing or unknown fields.
    */
  def parse(json: String, what: String = "columnar manifest"): ColumnarManifest =
    try parseManifest(JsonMethods.parse(json))
    catch {
      case e: ManifestException => fatal(s"$what: ${e.getMessage}")
      case e: com.fasterxml.jackson.core.JsonProcessingException =>
        fatal(s"$what: manifest is not valid JSON: ${e.getOriginalMessage}")
    }

  /** The fields of the JSON object `jv`, which must be exactly `names`. */
  private def fields(jv: JValue, where: String, names: String*): Map[String, JValue] = {
    val fs = jv match {
      case JObject(fs) => fs.toMap
      case _ => reject(s"'$where' is not an object")
    }
    names.find(!fs.contains(_)).foreach(n => reject(s"missing field '$n' in '$where'"))
    fs.keys.find(!names.contains(_)).foreach(n => reject(s"unknown field '$n' in '$where'"))
    fs
  }

  private def string(jv: JValue, where: String): String = jv match {
    case JString(s) => s
    case _ => reject(s"'$where' is not a string")
  }

  private def boolean(jv: JValue, where: String): Boolean = jv match {
    case JBool(b) => b
    case _ => reject(s"'$where' is not a boolean")
  }

  private def int(jv: JValue, where: String): Int = jv match {
    case JInt(i) if i.isValidInt => i.toInt
    case _ => reject(s"'$where' is not an int32")
  }

  private def array(jv: JValue, where: String): List[JValue] = jv match {
    case JArray(vs) => vs
    case _ => reject(s"'$where' is not an array")
  }

  private def schema(jv: JValue, where: String): Schema =
    try ColumnarFileSink.decodeSchema(string(jv, where))
    catch {
      case e: ManifestException => throw e
      case e: Exception => reject(s"'$where' is not a base64 Arrow IPC schema: ${e.getMessage}")
    }

  /** The path of a `{path, container}` file reference. */
  private def fileRef(jv: JValue, where: String): String = {
    val f = fields(jv, where, "path", "container")
    string(f("container"), s"$where.container") match {
      case Container => string(f("path"), s"$where.path")
      case c => reject(s"unsupported container '$c' in '$where'")
    }
  }

  private def parseManifest(jv: JValue): ColumnarManifest = {
    // Identify the format, version and kind before checking fields, which depend on them.
    val (format, version, kind) = jv match {
      case o: JObject => (o \ "format", o \ "version", o \ "kind")
      case _ => reject("not a JSON object")
    }
    if (format != JString(FormatName))
      reject(s"format is ${JsonMethods.compact(format)}, expected '$FormatName'")
    string(version, "version") match {
      case FormatVersion =>
      case v => reject(s"version '$v' is not supported; this reader supports '$FormatVersion'")
    }
    val isMatrix = string(kind, "kind") match {
      case "table" => false
      case "matrix_table" => true
      case k => reject(s"unknown kind '$k'")
    }

    val root = fields(
      jv,
      "manifest",
      Seq(
        "format",
        "version",
        "kind",
        "rows",
        "globals",
        "partitions",
        "internal_fields",
        "reference_genomes",
      ) ++ (if (isMatrix) Seq("cols", "entries") else Seq()): _*
    )

    val rows = fields(root("rows"), "rows", "schema", "key", "distinctly_keyed")
    val rowsSchema = schema(rows("schema"), "rows.schema")
    val key = array(rows("key"), "rows.key").map(string(_, "rows.key[]")).toIndexedSeq
    val distinctlyKeyed = boolean(rows("distinctly_keyed"), "rows.distinctly_keyed")

    val internal = fields(
      root("internal_fields"),
      "internal_fields",
      "row_id" +: (if (isMatrix) Seq("col_idx", "entry_valid") else Seq()): _*
    )
    val rowIdField = string(internal("row_id"), "internal_fields.row_id")

    val rowFields = rowsSchema.getFields.asScala
    checkRowIdField(rowsSchema, rowIdField, "rows.schema")
    val rowFieldNames = rowFields.tail.map(_.getName).toSet
    key.find(!rowFieldNames(_)).foreach(k => reject(s"key field '$k' is not a row field"))
    if (key.isEmpty && distinctlyKeyed)
      reject("an unkeyed table can't be distinctly_keyed")

    val globals = fields(root("globals"), "globals", "schema", "file")
    val globalsSchema = schema(globals("schema"), "globals.schema")
    val globalsFile = globals("file") match {
      case JNull => None
      case f => Some(fileRef(f, "globals.file"))
    }
    if (globalsFile.isEmpty != globalsSchema.getFields.isEmpty)
      reject("'globals.file' must be null exactly when the globals have no fields")

    val referenceGenomes =
      array(root("reference_genomes"), "reference_genomes").map { rg =>
        val parsed =
          try ReferenceGenome.parse(JsonMethods.compact(rg))
          catch {
            case e: Exception =>
              reject(s"invalid reference genome definition: ${e.getMessage}")
          }
        if (ReferenceGenome.builtinReferences().contains(parsed.name))
          reject(s"'reference_genomes' must not define the builtin reference genome " +
            s"'${parsed.name}'")
        parsed
      }.toIndexedSeq

    val matrix =
      if (isMatrix) Some(parseMatrixComponents(root, internal, rowIdField)) else None

    ColumnarManifest(
      rowsSchema = rowsSchema,
      key = key,
      distinctlyKeyed = distinctlyKeyed,
      globalsSchema = globalsSchema,
      globalsFile = globalsFile,
      partitionsFile = fileRef(root("partitions"), "partitions"),
      rowIdField = rowIdField,
      referenceGenomes = referenceGenomes,
      matrix = matrix,
    )
  }

  private def isNonNull(f: Field, t: ArrowType): Boolean =
    !f.isNullable && f.getType == t && f.getChildren.isEmpty

  private def checkRowIdField(schema: Schema, rowIdField: String, where: String): Unit =
    schema.getFields.asScala.headOption.filter(_.getName == rowIdField) match {
      case Some(f) if isNonNull(f, new ArrowType.Int(64, true)) =>
      case _ =>
        reject(s"the row ID field '$rowIdField' is not the first field of '$where', a " +
          "non-null int64")
    }

  /** The `cols` and `entries` fields of a matrix table's manifest, whose internal fields are
    * `internal`.
    */
  private def parseMatrixComponents(
    root: Map[String, JValue],
    internal: Map[String, JValue],
    rowIdField: String,
  ): ColumnarMatrixComponents = {
    val colIdxField = string(internal("col_idx"), "internal_fields.col_idx")
    val entryValidField = string(internal("entry_valid"), "internal_fields.entry_valid")

    val cols = fields(root("cols"), "cols", "schema", "key", "count", "file")
    val colsSchema = schema(cols("schema"), "cols.schema")
    val colKey = array(cols("key"), "cols.key").map(string(_, "cols.key[]")).toIndexedSeq
    val colCount = int(cols("count"), "cols.count")
    if (colCount < 0) reject(s"'cols.count' is negative: $colCount")
    val colFields = colsSchema.getFields.asScala
    colFields.headOption.filter(_.getName == colIdxField) match {
      case Some(f) if isNonNull(f, new ArrowType.Int(32, true)) =>
      case _ =>
        reject(s"the column index field '$colIdxField' is not the first field of 'cols.schema', " +
          "a non-null int32")
    }
    val colFieldNames = colFields.tail.map(_.getName).toSet
    colKey.find(!colFieldNames(_)).foreach(k =>
      reject(s"column key field '$k' is not a column field")
    )

    val entries = fields(root("entries"), "entries", "schema", "layout", "column_blocks")
    val entriesSchema = schema(entries("schema"), "entries.schema")
    string(entries("layout"), "entries.layout") match {
      case "dense" =>
      case l => reject(s"layout '$l' is not supported; this reader supports 'dense'")
    }
    val blocks = array(entries("column_blocks"), "entries.column_blocks").map { b =>
      val f = fields(b, "entries.column_blocks[]", "start", "end")
      (
        int(f("start"), "entries.column_blocks[].start"),
        int(f("end"), "entries.column_blocks[].end"),
      )
    }
    val covers = blocks.nonEmpty && blocks.head._1 == 0 && blocks.last._2 == colCount &&
      blocks.forall { case (s, e) => s <= e } &&
      blocks.zip(blocks.tail).forall { case ((_, e), (s, _)) => e == s }
    if (!covers)
      reject(s"'entries.column_blocks' must be contiguous ranges that cover the $colCount " +
        s"columns, found ${blocks.map { case (s, e) => s"[$s, $e)" }.mkString(", ")}")
    if (blocks.length != 1)
      reject(s"this reader supports only one column block, found ${blocks.length}")

    checkRowIdField(entriesSchema, rowIdField, "entries.schema")
    val entryFields = entriesSchema.getFields.asScala
    def isList(f: Field): Boolean =
      !f.isNullable && f.getType == ArrowType.List.INSTANCE && f.getChildren.size == 1
    entryFields.lift(1).filter(_.getName == entryValidField) match {
      case Some(f) if isList(f) && isNonNull(f.getChildren.get(0), ArrowType.Bool.INSTANCE) =>
      case _ =>
        reject(s"the entry validity field '$entryValidField' is not the second field of " +
          "'entries.schema', a non-null list of non-null booleans")
    }
    entryFields.drop(2).find(!isList(_)).foreach { f =>
      reject(s"entry field '${f.getName}' is not a non-null list in 'entries.schema'")
    }

    ColumnarMatrixComponents(
      colsSchema = colsSchema,
      colKey = colKey,
      colCount = colCount,
      colsFile = fileRef(cols("file"), "cols.file"),
      entriesSchema = entriesSchema,
      colIdxField = colIdxField,
      entryValidField = entryValidField,
    )
  }
}
