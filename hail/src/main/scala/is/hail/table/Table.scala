package is.hail.table

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.rvd._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.{DataFrame, Row}
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

sealed abstract class SortOrder

case object Ascending extends SortOrder

case object Descending extends SortOrder

case class SortField(field: String, sortOrder: SortOrder)

abstract class AbstractTableSpec extends RelationalSpec {
  def references_rel_path: String
  def table_type: TableType
  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")
  def rvdType(path: String): RVDType = {
    val rows = AbstractRVDSpec.read(HailContext.get, path + "/" + rowsComponent.rel_path)
    RVDType(rows.encodedType, rows.key)
  }
}

case class TableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends AbstractTableSpec

object Table {
  def range(hc: HailContext, n: Int, nPartitions: Option[Int] = None): TableIR =
    TableRange(n, nPartitions.getOrElse(hc.sc.defaultParallelism))

  def pyFromDF(df: DataFrame, jKey: java.util.ArrayList[String]): TableIR = {
    val key = jKey.asScala.toArray.toFastIndexedSeq
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    TableLiteral(TableValue(signature, key, df.rdd))
  }

  def read(hc: HailContext, path: String): TableIR =
    TableIR.read(hc, path, dropRows = false, None)

  def importFamJSON(path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): String = {
    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)
    val (data, typ) = LoadPlink.parseFam(path, ffConfig, HailContext.get.hadoopConf)
    val jv = JSONAnnotationImpex.exportAnnotation(
      Row(typ.toString, data),
      TStruct("type" -> TString(), "data" -> TArray(typ)))
    JsonMethods.compact(jv)
  }
}
