package is.hail.compatibility

import is.hail.expr.types.{MatrixType, TableType}
import is.hail.table.TableSpec
import is.hail.utils.SemanticVersion
import is.hail.variant._
import org.json4s.JsonAST.JValue
import org.json4s.{Formats, ShortTypeHints}

case class MatrixTableSpec_1_0(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  matrix_type: MatrixType,
  components: Map[String, ComponentSpec]) extends MatrixTableSpec

case class TableSpec_1_0(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends TableSpec

object BackCompatibleReader {
  def extract(fileVersion: SemanticVersion, jv: JValue): RelationalSpec = {
    implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(List(classOf[MatrixTableSpec_1_0], classOf[TableSpec_1_0]))
    fileVersion match {
      case SemanticVersion(1, 0, _) =>
        jv.transformField { case ("MatrixTableSpec", value) => ("MatrixTableSpec_1_0", value) }
          .transformField { case ("TableSpec", value) => ("TableSpec_1_0", value) }
          .extract[RelationalSpec]
    }
  }
}
