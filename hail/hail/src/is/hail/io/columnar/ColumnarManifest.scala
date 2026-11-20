package is.hail.io.columnar

import is.hail.io.fs.FS
import is.hail.utils.{fatal, using}
import is.hail.variant.ReferenceGenome

import scala.io.Source
import scala.jdk.CollectionConverters._

import java.nio.charset.StandardCharsets

import org.apache.arrow.vector.types.pojo.{ArrowType, Schema}
import org.json4s.{JArray, JBool, JNull, JObject, JString, JValue}
import org.json4s.jackson.JsonMethods

/** A table's `manifest.json`, as a reader sees it (spec section 3). Paths are relative to the
  * table's directory.
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
)

object ColumnarManifest {
  import RelationalColumnarWriter._

  /** Reads and validates `<path>/manifest.json`. A directory without one is not a committed table
    * (spec section 2.1).
    */
  def read(fs: FS, path: String): ColumnarManifest = {
    val manifestPath = s"$path/$ManifestFile"
    if (!fs.isFile(manifestPath))
      fatal(s"'$path' is not a columnar table: it has no $ManifestFile")
    val json = using(Source.fromInputStream(fs.open(manifestPath), StandardCharsets.UTF_8.name))(
      _.mkString
    )
    parse(json, s"columnar table '$path'")
  }

  private class ManifestException(msg: String) extends RuntimeException(msg)

  private def reject(msg: String): Nothing = throw new ManifestException(msg)

  /** Parses and validates the contents of a `manifest.json`. Fails on anything a `0.x` reader must
    * reject: another format, version, kind or container, and missing or unknown fields.
    */
  def parse(json: String, what: String = "columnar table manifest"): ColumnarManifest =
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
    string(kind, "kind") match {
      case "table" =>
      case "matrix_table" => reject("it is a matrix table, not a table")
      case k => reject(s"unknown kind '$k'")
    }

    val root = fields(
      jv,
      "manifest",
      "format",
      "version",
      "kind",
      "rows",
      "globals",
      "partitions",
      "internal_fields",
      "reference_genomes",
    )

    val rows = fields(root("rows"), "rows", "schema", "key", "distinctly_keyed")
    val rowsSchema = schema(rows("schema"), "rows.schema")
    val key = array(rows("key"), "rows.key").map(string(_, "rows.key[]")).toIndexedSeq
    val distinctlyKeyed = boolean(rows("distinctly_keyed"), "rows.distinctly_keyed")

    val internal = fields(root("internal_fields"), "internal_fields", "row_id")
    val rowIdField = string(internal("row_id"), "internal_fields.row_id")

    val rowFields = rowsSchema.getFields.asScala
    rowFields.headOption.filter(_.getName == rowIdField) match {
      case Some(f) if !f.isNullable && f.getType == new ArrowType.Int(64, true) =>
      case _ =>
        reject(s"the row ID field '$rowIdField' is not the first field of 'rows.schema', a " +
          "non-null int64")
    }
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

    ColumnarManifest(
      rowsSchema = rowsSchema,
      key = key,
      distinctlyKeyed = distinctlyKeyed,
      globalsSchema = globalsSchema,
      globalsFile = globalsFile,
      partitionsFile = fileRef(root("partitions"), "partitions"),
      rowIdField = rowIdField,
      referenceGenomes = referenceGenomes,
    )
  }
}
