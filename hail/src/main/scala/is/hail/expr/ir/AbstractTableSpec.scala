package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.types._
import is.hail.rvd._

import scala.language.implicitConversions

object SortOrder {
  def deserialize(b: Byte): SortOrder =
    if (b == 0.toByte) Ascending
    else if (b == 1.toByte) Descending
    else throw new RuntimeException(s"invalid sort order: $b")
}

sealed abstract class SortOrder {
  def serialize: Byte
}

case object Ascending extends SortOrder {
  def serialize: Byte = 0.toByte
}

case object Descending extends SortOrder {
  def serialize: Byte = 1.toByte
}

case class SortField(field: String, sortOrder: SortOrder)

abstract class AbstractTableSpec extends RelationalSpec {
  def references_rel_path: String
  def table_type: TableType
  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def rowsSpec(path: String): AbstractRVDSpec = AbstractRVDSpec.read(HailContext.get, path + "/" + rowsComponent.rel_path)
  def indexed(path: String): Boolean = rowsSpec(path).indexed
}

case class TableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends AbstractTableSpec
