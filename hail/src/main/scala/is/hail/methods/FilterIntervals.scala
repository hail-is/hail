package is.hail.methods

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, TableToTableFunction}
import is.hail.expr.ir.{IRParser, MatrixValue, TableValue}
import is.hail.expr.types.virtual.{TArray, TInterval, Type}
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.utils.Interval
import org.json4s.CustomSerializer
import org.json4s.JsonAST.{JBool, JObject, JString}

class MatrixFilterIntervalsSerializer extends CustomSerializer[MatrixFilterIntervals](format => (
  { case JObject(fields) =>
      val fieldMap = fields.toMap

      val keyType = IRParser.parseType(fieldMap("keyType").asInstanceOf[JString].s)
      val intervals = JSONAnnotationImpex.importAnnotation(fieldMap("intervals"), TArray(TInterval(keyType)))
      MatrixFilterIntervals(
        keyType,
        intervals.asInstanceOf[IndexedSeq[Interval]].toFastIndexedSeq,
        fieldMap("keep").asInstanceOf[JBool].value)
  },
  { case fi: MatrixFilterIntervals =>
      JObject("name" -> JString("MatrixFilterIntervals"),
        "keyType" -> JString(fi.keyType.parsableString()),
        "intervals" -> JSONAnnotationImpex.exportAnnotation(fi.intervals.toFastIndexedSeq, TArray(TInterval(fi.keyType))),
        "keep" -> JBool(fi.keep))
  }))

case class MatrixFilterIntervals(
  keyType: Type,
  intervals: IndexedSeq[Interval],
  keep: Boolean) extends MatrixToMatrixFunction {
  def preservesPartitionCounts: Boolean = false

  def typeInfo(childType: MatrixType, childRVDType: RVDType): (MatrixType, RVDType) = {
    (childType, childRVDType)
  }

  override def lower(): Option[TableToTableFunction] = Some(TableFilterIntervals(keyType, intervals, keep))

  def execute(mv: MatrixValue): MatrixValue = throw new UnsupportedOperationException
}

class TableFilterIntervalsSerializer extends CustomSerializer[TableFilterIntervals](format => (
  { case JObject(fields) =>
    val fieldMap = fields.toMap

    val keyType = IRParser.parseType(fieldMap("keyType").asInstanceOf[JString].s)
    val intervals = JSONAnnotationImpex.importAnnotation(fieldMap("intervals"), TArray(TInterval(keyType)))
    TableFilterIntervals(
      keyType,
      intervals.asInstanceOf[IndexedSeq[Interval]].toFastIndexedSeq,
      fieldMap("keep").asInstanceOf[JBool].value)
  },
  { case fi: TableFilterIntervals =>
    JObject("name" -> JString("TableFilterIntervals"),
      "keyType" -> JString(fi.keyType.parsableString()),
      "intervals" -> JSONAnnotationImpex.exportAnnotation(fi.intervals.toFastIndexedSeq, TArray(TInterval(fi.keyType))),
      "keep" -> JBool(fi.keep))
  }))

case class TableFilterIntervals(
  keyType: Type,
  intervals: IndexedSeq[Interval],
  keep: Boolean) extends TableToTableFunction {
  def preservesPartitionCounts: Boolean = false

  def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType) = {
    (childType, childRVDType)
  }

  def execute(tv: TableValue): TableValue = {
    val partitioner = RVDPartitioner.union(
      tv.typ.keyType,
      intervals,
      tv.rvd.typ.key.length - 1)
    TableValue(tv.typ, tv.globals, tv.rvd.filterIntervals(partitioner, keep))
  }
}
