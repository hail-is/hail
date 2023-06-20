package is.hail.expr.ir.functions

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.MonadLower
import is.hail.expr.ir.{MatrixValue, RelationalSpec, TableValue}
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.types.virtual.Type
import is.hail.types._
import is.hail.utils._
import org.json4s.jackson.JsonMethods
import org.json4s.{Extraction, JValue, ShortTypeHints}

import scala.language.higherKinds

abstract class MatrixToMatrixFunction {
  def typ(childType: MatrixType): MatrixType

  def preservesPartitionCounts: Boolean

  def lower(): TableToTableFunction

  def requestType(requestedType: MatrixType, childBaseType: MatrixType): MatrixType = childBaseType
}

abstract class MatrixToTableFunction {
  def typ(childType: MatrixType): TableType

  def execute[M[_]: MonadLower](mv: MatrixValue): M[TableValue]

  def preservesPartitionCounts: Boolean

  def lower(): Option[TableToTableFunction] = None
}

abstract class BlockMatrixToTableFunction {
  def typ(bmType: BlockMatrixType, auxType: Type): TableType

  def execute[M[_]: MonadLower](bm: BlockMatrix, aux: Any): M[TableValue]
}

case class WrappedMatrixToTableFunction(
  function: MatrixToTableFunction,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableToTableFunction {
  override def typ(childType: TableType): TableType = {
    val mType = MatrixType.fromTableType(childType, colsFieldName, entriesFieldName, colKey)
    function.typ(mType) // MatrixType RVDTypes will go away
  }

  def execute[M[_]: MonadLower](tv: TableValue): M[TableValue] =
    function.execute(tv.toMatrixValue(colKey, colsFieldName, entriesFieldName))

  override def preservesPartitionCounts: Boolean = function.preservesPartitionCounts
}

abstract class TableToTableFunction {
  def typ(childType: TableType): TableType

  def execute[M[_]: MonadLower](tv: TableValue): M[TableValue]

  def preservesPartitionCounts: Boolean

  def requestType(requestedType: TableType, childBaseType: TableType): TableType = childBaseType

  def toJValue: JValue = {
    Extraction.decompose(this)(RelationalFunctions.formats)
  }
}

abstract class TableToValueFunction {
  def typ(childType: TableType): Type

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit

  def execute[M[_]: MonadLower](tv: TableValue): M[Any]
}

case class WrappedMatrixToValueFunction(
  function: MatrixToValueFunction,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableToValueFunction {

  def typ(childType: TableType): Type = {
    function.typ(MatrixType.fromTableType(childType, colsFieldName, entriesFieldName, colKey))
  }

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = function.unionRequiredness(childType, resultType)

  override def execute[M[_]: MonadLower](tv: TableValue): M[Any] =
    function.execute(tv.toMatrixValue(colKey, colsFieldName, entriesFieldName))
}

abstract class MatrixToValueFunction {
  def typ(childType: MatrixType): Type

  def execute[M[_]: MonadLower](mv: MatrixValue): M[Any]

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit

  def lower(): Option[TableToValueFunction] = None
}

abstract class BlockMatrixToValueFunction {
  def typ(childType: BlockMatrixType): Type

  def execute[M[_]: MonadLower](bm: BlockMatrix): M[Any]
}

object RelationalFunctions {
  implicit val formats = RelationalSpec.formats + ShortTypeHints(List(
    classOf[LinearRegressionRowsSingle],
    classOf[LinearRegressionRowsChained],
    classOf[TableFilterPartitions],
    classOf[MatrixFilterPartitions],
    classOf[TableCalculateNewPartitions],
    classOf[ForceCountTable],
    classOf[ForceCountMatrixTable],
    classOf[NPartitionsTable],
    classOf[NPartitionsMatrixTable],
    classOf[LogisticRegression],
    classOf[PoissonRegression],
    classOf[Skat],
    classOf[LocalLDPrune],
    classOf[MatrixExportEntriesByCol],
    classOf[PCA],
    classOf[VEP],
    classOf[IBD],
    classOf[Nirvana],
    classOf[GetElement],
    classOf[WrappedMatrixToTableFunction],
    classOf[WrappedMatrixToValueFunction],
    classOf[PCRelate]
  ), typeHintFieldName = "name")

  def extractTo[T: Manifest](ctx: ExecuteContext, config: String): T = {
    val jv = JsonMethods.parse(config)
    (jv \ "name").extract[String] match {
      case "VEP" => VEP.fromJValue(ctx.fs, jv).asInstanceOf[T]
      case _ => {
        log.info("JSON: " + jv.toString)
        jv.extract[T]
      }
    }
  }

  def lookupMatrixToMatrix(ctx: ExecuteContext, config: String): MatrixToMatrixFunction = extractTo[MatrixToMatrixFunction](ctx, config)

  def lookupMatrixToTable(ctx: ExecuteContext, config: String): MatrixToTableFunction = extractTo[MatrixToTableFunction](ctx, config)

  def lookupTableToTable(ctx: ExecuteContext, config: String): TableToTableFunction = extractTo[TableToTableFunction](ctx, config)

  def lookupBlockMatrixToTable(ctx: ExecuteContext, config: String): BlockMatrixToTableFunction = extractTo[BlockMatrixToTableFunction](ctx, config)

  def lookupTableToValue(ctx: ExecuteContext, config: String): TableToValueFunction = extractTo[TableToValueFunction](ctx, config)

  def lookupMatrixToValue(ctx: ExecuteContext, config: String): MatrixToValueFunction = extractTo[MatrixToValueFunction](ctx, config)

  def lookupBlockMatrixToValue(ctx: ExecuteContext, config: String): BlockMatrixToValueFunction = extractTo[BlockMatrixToValueFunction](ctx, config)
}
