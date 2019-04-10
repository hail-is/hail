package is.hail.expr.ir.functions

import is.hail.expr.ir.{LowerMatrixIR, MatrixValue, TableValue}
import is.hail.expr.types.virtual.Type
import is.hail.expr.types.{BlockMatrixType, MatrixType, TableType}
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.rvd.RVDType
import is.hail.variant.RelationalSpec
import org.json4s.ShortTypeHints
import org.json4s.jackson.Serialization


abstract class MatrixToMatrixFunction {
  def typeInfo(childType: MatrixType, childRVDType: RVDType): (MatrixType, RVDType)

  def execute(mv: MatrixValue): MatrixValue

  def preservesPartitionCounts: Boolean

  def lower(): Option[TableToTableFunction] = None
}

abstract class MatrixToTableFunction {
  def typeInfo(childType: MatrixType, childRVDType: RVDType): (TableType, RVDType)

  def execute(mv: MatrixValue): TableValue

  def preservesPartitionCounts: Boolean

  def lower(): Option[TableToTableFunction] = None
}

case class WrappedMatrixToTableFunction(
  function: MatrixToTableFunction,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableToTableFunction {
  override def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType) = {
    val mType = MatrixType.fromTableType(childType, colsFieldName, entriesFieldName, colKey)
    function.typeInfo(mType, mType.canonicalRVDType) // MatrixType RVDTypes will go away
  }

  def execute(tv: TableValue): TableValue = function.execute(tv.toMatrixValue(colsFieldName, entriesFieldName, colKey))

  override def preservesPartitionCounts: Boolean = function.preservesPartitionCounts
}

case class WrappedMatrixToMatrixFunction(function: MatrixToMatrixFunction,
  inColsFieldName: String,
  outColsFieldName: String,
  inEntriesFieldName: String,
  outEntriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableToTableFunction {
  override def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType) = {
    val mType = MatrixType.fromTableType(childType, inColsFieldName, inEntriesFieldName, colKey)
    val (outMatrixType, _) = function.typeInfo(mType, mType.canonicalRVDType) // MatrixType RVDTypes will go away

    val tt = LowerMatrixIR.loweredType(outMatrixType, outEntriesFieldName, outColsFieldName)
    tt -> tt.canonicalRVDType
  }

  def execute(tv: TableValue): TableValue = function.execute(tv
    .toMatrixValue(inColsFieldName, inEntriesFieldName, colKey))
    .toTableValue(outColsFieldName, outEntriesFieldName)

  def preservesPartitionCounts: Boolean = function.preservesPartitionCounts
}

abstract class TableToTableFunction {
  def typeInfo(childType: TableType, childRVDType: RVDType): (TableType, RVDType)

  def execute(tv: TableValue): TableValue

  def preservesPartitionCounts: Boolean
}

abstract class TableToValueFunction {
  def typ(childType: TableType): Type

  def execute(tv: TableValue): Any
}

case class WrappedMatrixToValueFunction(
  function: MatrixToValueFunction,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableToValueFunction {

  def typ(childType: TableType): Type = {
    function.typ(MatrixType.fromTableType(childType, colsFieldName, entriesFieldName, colKey))
  }

  def execute(tv: TableValue): Any = function.execute(tv.toMatrixValue(colsFieldName, entriesFieldName, colKey))
}

abstract class MatrixToValueFunction {
  def typ(childType: MatrixType): Type

  def execute(mv: MatrixValue): Any

  def lower(): Option[TableToValueFunction] = None
}

abstract class BlockMatrixToValueFunction {
  def typ(childType: BlockMatrixType): Type

  def execute(bm: BlockMatrix): Any
}

object RelationalFunctions {
  implicit val formats = RelationalSpec.formats + ShortTypeHints(List(
    classOf[LinearRegressionRowsSingle],
    classOf[LinearRegressionRowsChained],
    classOf[WindowByLocus],
    classOf[TableFilterPartitions],
    classOf[MatrixFilterPartitions],
    classOf[ForceCountTable],
    classOf[ForceCountMatrixTable],
    classOf[NPartitionsTable],
    classOf[NPartitionsMatrixTable],
    classOf[LogisticRegression],
    classOf[MatrixWriteBlockMatrix],
    classOf[TableFilterIntervals],
    classOf[MatrixFilterIntervals],
    classOf[PoissonRegression],
    classOf[Skat],
    classOf[LocalLDPrune],
    classOf[MatrixExportEntriesByCol],
    classOf[PCA],
    classOf[VEP],
    classOf[GetElement],
    classOf[WrappedMatrixToTableFunction],
    classOf[WrappedMatrixToMatrixFunction],
    classOf[WrappedMatrixToValueFunction]
  )) +
    new MatrixFilterIntervalsSerializer +
    new TableFilterIntervalsSerializer

  def extractTo[T : Manifest](config: String): T = {
    Serialization.read[T](config)
  }

  def lookupMatrixToMatrix(config: String): MatrixToMatrixFunction = extractTo[MatrixToMatrixFunction](config)
  def lookupMatrixToTable(config: String): MatrixToTableFunction = extractTo[MatrixToTableFunction](config)
  def lookupTableToTable(config: String): TableToTableFunction = extractTo[TableToTableFunction](config)
  def lookupTableToValue(config: String): TableToValueFunction = extractTo[TableToValueFunction](config)
  def lookupMatrixToValue(config: String): MatrixToValueFunction = extractTo[MatrixToValueFunction](config)
  def lookupBlockMatrixToValue(config: String): BlockMatrixToValueFunction = extractTo[BlockMatrixToValueFunction](config)
}
