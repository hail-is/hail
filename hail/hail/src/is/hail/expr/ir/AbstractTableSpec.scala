package is.hail.expr.ir

import is.hail.io.fs.FS
import is.hail.rvd._
import is.hail.types.virtual.TableType
import is.hail.utils._

import java.io.OutputStreamWriter

import org.json4s.{Formats, JValue}
import org.json4s.jackson.JsonMethods

object SortOrder {
  def deserialize(b: Byte): SortOrder =
    if (b == 0.toByte) Ascending
    else if (b == 1.toByte) Descending
    else throw new RuntimeException(s"invalid sort order: $b")

  def parse(s: String): SortOrder = s match {
    case "A" => Ascending
    case "D" => Descending
    case x => throw new RuntimeException(s"invalid sort order: $x")
  }
}

sealed abstract class SortOrder {
  def serialize: Byte
  def parsableString(): String
}

case object Ascending extends SortOrder {
  override def serialize: Byte = 0.toByte
  override def parsableString(): String = "A"
}

case object Descending extends SortOrder {
  override def serialize: Byte = 1.toByte
  override def parsableString(): String = "D"
}

object SortField {
  def parse(s: String): SortField =
    IRParser.parseSortField(s)
}

case class SortField(field: String, sortOrder: SortOrder) {
  def parsableString(): String = sortOrder.parsableString() + field
  override def toString(): String = parsableString()
}

abstract class AbstractTableSpec extends RelationalSpec {
  def references_rel_path: String

  def table_type: TableType

  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def rowsSpec: AbstractRVDSpec

  def globalsSpec: AbstractRVDSpec

  override def indexed: Boolean = rowsSpec.indexed
}

object TableSpec {
  def apply(fs: FS, path: String, params: TableSpecParameters): TableSpec = {
    val globalsComponent = params.components("globals").asInstanceOf[RVDComponentSpec]
    val globalsSpec = globalsComponent.rvdSpec(fs, path)

    val rowsComponent = params.components("rows").asInstanceOf[RVDComponentSpec]
    val rowsSpec = rowsComponent.rvdSpec(fs, path)

    new TableSpec(params, globalsSpec, rowsSpec)
  }

  def fromJValue(fs: FS, path: String, jv: JValue): TableSpec = {
    implicit val formats: Formats = RelationalSpec.formats
    val params = jv.extract[TableSpecParameters]
    TableSpec(fs, path, params)
  }
}

case class TableSpecParameters(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec],
) {

  def write(fs: FS, path: String): Unit =
    using(new OutputStreamWriter(fs.create(path + "/metadata.json.gz"))) { out =>
      out.write(JsonMethods.compact(decomposeWithName(this, "TableSpec")(RelationalSpec.formats)))
    }
}

class TableSpec(
  val params: TableSpecParameters,
  val globalsSpec: AbstractRVDSpec,
  val rowsSpec: AbstractRVDSpec,
) extends AbstractTableSpec {
  override def file_version: Int = params.file_version

  override def hail_version: String = params.hail_version

  override def components: Map[String, ComponentSpec] = params.components

  override def references_rel_path: String = params.references_rel_path

  override def table_type: TableType = params.table_type

  override def toJValue: JValue =
    decomposeWithName(params, "TableSpec")(RelationalSpec.formats)
}
