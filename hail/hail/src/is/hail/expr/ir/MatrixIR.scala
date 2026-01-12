package is.hail.expr.ir

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.DeprecatedIRBuilder._
import is.hail.expr.ir.analyses.{ColumnCount, PartitionCounts}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.MatrixToMatrixFunction
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.fs.FS
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.rvd._
import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils._

import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods

object MatrixIR {
  def read(
    fs: FS,
    path: String,
    dropCols: Boolean = false,
    dropRows: Boolean = false,
    requestedType: Option[MatrixType] = None,
  ): MatrixIR = {
    val reader = MatrixNativeReader(fs, path)
    MatrixRead(requestedType.getOrElse(reader.fullMatrixType), dropCols, dropRows, reader)
  }

  def range(
    ctx: ExecuteContext,
    nRows: Int,
    nCols: Int,
    nPartitions: Option[Int],
    dropCols: Boolean = false,
    dropRows: Boolean = false,
  ): MatrixIR = {
    val reader = MatrixRangeReader(ctx, nRows, nCols, nPartitions)
    val requestedType = reader.fullMatrixTypeWithoutUIDs
    MatrixRead(requestedType, dropCols = dropCols, dropRows = dropRows, reader = reader)
  }

  val globalName: Name = Name("global")
  val rowName: Name = Name("va")
  val colName: Name = Name("sa")
  val entryName: Name = Name("g")
}

sealed abstract class MatrixIR extends BaseIR {
  override def typ: MatrixType

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixIR
}

object MatrixLiteral {
  def apply(
    ctx: ExecuteContext,
    typ: MatrixType,
    rvd: RVD,
    globals: Row,
    colValues: IndexedSeq[Row],
  ): MatrixLiteral = {
    val tt = typ.canonicalTableType
    MatrixLiteral(
      typ,
      TableLiteral(
        TableValue(
          ctx,
          tt,
          BroadcastRow(
            ctx,
            Row.fromSeq(globals.toSeq :+ colValues),
            typ.canonicalTableType.globalType,
          ),
          rvd,
        ),
        ctx.theHailClassLoader,
      ),
    )
  }
}

case class MatrixLiteral(typ: MatrixType, tl: TableLiteral) extends MatrixIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    MatrixLiteral(typ, tl)
  }

  override def toString: String = "MatrixLiteral(...)"
}

object MatrixReader {
  def fromJson(ctx: ExecuteContext, jv: JValue): MatrixReader = {
    implicit val formats: Formats = DefaultFormats
    (jv \ "name").extract[String] match {
      case "MatrixRangeReader" => MatrixRangeReader.fromJValue(ctx, jv)
      case "MatrixNativeReader" => MatrixNativeReader.fromJValue(ctx.fs, jv)
      case "MatrixBGENReader" => MatrixBGENReader.fromJValue(ctx, jv)
      case "MatrixPLINKReader" => MatrixPLINKReader.fromJValue(ctx, jv)
      case "MatrixVCFReader" => MatrixVCFReader.fromJValue(ctx, jv)
    }
  }

  val rowUIDFieldName: String = TableReader.uidFieldName

  val colUIDFieldName: String = "__col_uid"
}

trait MatrixReader {
  def pathsUsed: Seq[String]

  def columnCount: Option[Int]

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullMatrixTypeWithoutUIDs: MatrixType

  def rowUIDType: Type

  def colUIDType: Type

  lazy val fullMatrixType: MatrixType = {
    val mt = fullMatrixTypeWithoutUIDs
    val rowType = mt.rowType
    val newRowType = if (rowType.hasField(rowUIDFieldName))
      rowType
    else
      rowType.appendKey(rowUIDFieldName, rowUIDType)
    val colType = mt.colType
    val newColType = if (colType.hasField(colUIDFieldName))
      colType
    else
      colType.appendKey(colUIDFieldName, colUIDType)

    mt.copy(rowType = newRowType, colType = newColType)
  }

  def lower(ctx: ExecuteContext, requestedType: MatrixType, dropCols: Boolean, dropRows: Boolean)
    : TableIR

  def toJValue: JValue

  def renderShort(): String

  def defaultRender(): String =
    StringEscapeUtils.escapeString(JsonMethods.compact(toJValue))

  final def matrixToTableType(mt: MatrixType, includeColsArray: Boolean = true): TableType = {
    TableType(
      rowType = if (mt.rowType.hasField(rowUIDFieldName))
        mt.rowType.deleteKey(rowUIDFieldName)
          .appendKey(LowerMatrixIR.entriesFieldName, TArray(mt.entryType))
          .appendKey(TableReader.uidFieldName, mt.rowType.fieldType(rowUIDFieldName))
      else
        mt.rowType.appendKey(LowerMatrixIR.entriesFieldName, TArray(mt.entryType)),
      key = mt.rowKey,
      globalType = if (includeColsArray)
        mt.globalType.appendKey(LowerMatrixIR.colsFieldName, TArray(mt.colType))
      else
        mt.globalType,
    )
  }

  final def rowUIDFieldName: String = MatrixReader.rowUIDFieldName

  final def colUIDFieldName: String = MatrixReader.colUIDFieldName
}

abstract class MatrixHybridReader extends TableReaderWithExtraUID with MatrixReader {
  override def uidType = rowUIDType

  override def fullTypeWithoutUIDs: TableType = matrixToTableType(
    fullMatrixTypeWithoutUIDs.copy(
      colType = fullMatrixTypeWithoutUIDs.colType.appendKey(colUIDFieldName, colUIDType)
    )
  )

  override def defaultRender(): String = super.defaultRender()

  override def lower(
    ctx: ExecuteContext,
    requestedType: MatrixType,
    dropCols: Boolean,
    dropRows: Boolean,
  ): TableIR = {
    var tr: TableIR = TableRead(matrixToTableType(requestedType), dropRows, this)
    if (dropCols) {
      // this lowering preserves dropCols using pruning
      tr = TableMapRows(
        tr,
        InsertFields(
          Ref(TableIR.rowName, tr.typ.rowType),
          FastSeq(LowerMatrixIR.entriesFieldName -> MakeArray(
            FastSeq(),
            TArray(requestedType.entryType),
          )),
        ),
      )
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref(TableIR.globalName, tr.typ.globalType),
          FastSeq(LowerMatrixIR.colsFieldName -> MakeArray(
            FastSeq(),
            TArray(requestedType.colType),
          )),
        ),
      )
    }
    tr
  }
}

object MatrixNativeReader {
  def apply(fs: FS, path: String, options: Option[NativeReaderOptions] = None): MatrixNativeReader =
    MatrixNativeReader(fs, MatrixNativeReaderParameters(path, options))

  def apply(fs: FS, params: MatrixNativeReaderParameters): MatrixNativeReader = {
    val spec =
      (RelationalSpec.read(fs, params.path): @unchecked) match {
        case mts: AbstractMatrixTableSpec => mts
        case _: AbstractTableSpec => fatal(s"file is a Table, not a MatrixTable: '${params.path}'")
      }

    val intervals = params.options.map(_.intervals)
    if (intervals.nonEmpty && !spec.indexed)
      fatal("""`intervals` specified on an unindexed matrix table.
              |This matrix table was written using an older version of hail
              |rewrite the matrix in order to create an index to proceed""".stripMargin)

    new MatrixNativeReader(params, spec)
  }

  def fromJValue(fs: FS, jv: JValue): MatrixNativeReader = {
    val path = jv \ "path" match {
      case JString(s) => s
    }

    val options = jv \ "options" match {
      case optionsJV: JObject =>
        Some(NativeReaderOptions.fromJValue(optionsJV))
      case JNothing => None
    }

    MatrixNativeReader(fs, MatrixNativeReaderParameters(path, options))
  }
}

case class MatrixNativeReaderParameters(
  path: String,
  options: Option[NativeReaderOptions],
)

class MatrixNativeReader(
  val params: MatrixNativeReaderParameters,
  spec: AbstractMatrixTableSpec,
) extends MatrixReader {
  override def pathsUsed: Seq[String] = FastSeq(params.path)

  override def renderShort(): String =
    s"(MatrixNativeReader ${params.path} ${params.options.map(_.renderShort()).getOrElse("")})"

  lazy val columnCount: Option[Int] = Some(spec.colsSpec
    .partitionCounts
    .sum
    .toInt)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (params.options.isEmpty) Some(spec.partitionCounts) else None

  override def fullMatrixTypeWithoutUIDs: MatrixType = spec.matrix_type

  override def rowUIDType = TTuple(TInt64, TInt64)
  override def colUIDType = TTuple(TInt64, TInt64)

  override def lower(
    ctx: ExecuteContext,
    requestedType: MatrixType,
    dropCols: Boolean,
    dropRows: Boolean,
  ): TableIR = {
    val rowsPath = params.path + "/rows"
    val entriesPath = params.path + "/entries"
    val colsPath = params.path + "/cols"

    if (dropCols) {
      val tt = TableType(requestedType.rowType, requestedType.rowKey, requestedType.globalType)
      val trdr: TableReader =
        new TableNativeReader(TableNativeReaderParameters(rowsPath, params.options), spec.rowsSpec)
      var tr: TableIR = TableRead(tt, dropRows, trdr)
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref(TableIR.globalName, tr.typ.globalType),
          FastSeq(LowerMatrixIR.colsFieldName -> MakeArray(
            FastSeq(),
            TArray(requestedType.colType),
          )),
        ),
      )
      TableMapRows(
        tr,
        InsertFields(
          Ref(TableIR.rowName, tr.typ.rowType),
          FastSeq(LowerMatrixIR.entriesFieldName -> MakeArray(
            FastSeq(),
            TArray(requestedType.entryType),
          )),
        ),
      )
    } else {
      val tt = matrixToTableType(requestedType, includeColsArray = false)
      val trdr = TableNativeZippedReader(
        rowsPath,
        entriesPath,
        params.options,
        spec.rowsSpec,
        spec.entriesSpec,
      )
      val tr: TableIR = TableRead(tt, dropRows, trdr)
      val colsRVDSpec = spec.colsSpec.rowsSpec
      val partFiles =
        colsRVDSpec.absolutePartPaths(spec.colsSpec.rowsComponent.absolutePath(colsPath))

      val cols = if (partFiles.length == 1) {
        ReadPartition(
          MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> Str(partFiles.head))),
          requestedType.colType,
          PartitionNativeReader(colsRVDSpec.typedCodecSpec, colUIDFieldName),
        )
      } else {
        val contextType = TStruct("partitionIndex" -> TInt64, "partitionPath" -> TString)
        val partNames = MakeArray(
          partFiles.zipWithIndex.map { case (path, idx) =>
            MakeStruct(Array("partitionIndex" -> I64(idx.toLong), "partitionPath" -> Str(path)))
          },
          TArray(contextType),
        )
        val elt = Ref(freshName(), contextType)
        StreamFlatMap(
          partNames,
          elt.name,
          ReadPartition(
            elt,
            requestedType.colType,
            PartitionNativeReader(colsRVDSpec.typedCodecSpec, colUIDFieldName),
          ),
        )
      }

      TableMapGlobals(
        tr,
        InsertFields(
          Ref(TableIR.globalName, tr.typ.globalType),
          FastSeq(LowerMatrixIR.colsFieldName -> ToArray(cols)),
        ),
      )
    }
  }

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixNativeReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixNativeReader => params == that.params
    case _ => false
  }

  def getSpec(): AbstractMatrixTableSpec = this.spec
}

object MatrixRangeReader {
  def apply(ctx: ExecuteContext, nRows: Int, nCols: Int, nPartitions: Option[Int])
    : MatrixRangeReader =
    MatrixRangeReader(ctx, MatrixRangeReaderParameters(nRows, nCols, nPartitions))

  def fromJValue(ctx: ExecuteContext, jv: JValue): MatrixRangeReader = {
    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[MatrixRangeReaderParameters]
    MatrixRangeReader(ctx, params)
  }

  def apply(ctx: ExecuteContext, params: MatrixRangeReaderParameters): MatrixRangeReader = {
    val nPartitionsAdj =
      math.min(params.nRows, params.nPartitions.getOrElse(ctx.backend.defaultParallelism))
    new MatrixRangeReader(params, nPartitionsAdj)
  }
}

case class MatrixRangeReaderParameters(nRows: Int, nCols: Int, nPartitions: Option[Int])

case class MatrixRangeReader(
  val params: MatrixRangeReaderParameters,
  nPartitionsAdj: Int,
) extends MatrixReader {
  override def pathsUsed: Seq[String] = FastSeq()

  override def rowUIDType = TInt64
  override def colUIDType = TInt64

  override def fullMatrixTypeWithoutUIDs: MatrixType = MatrixType(
    globalType = TStruct.empty,
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32),
    entryType = TStruct.empty,
  )

  override def renderShort(): String = s"(MatrixRangeReader $params $nPartitionsAdj)"

  val columnCount: Option[Int] = Some(params.nCols)

  lazy val partitionCounts: Option[IndexedSeq[Long]] =
    Some(partition(params.nRows, nPartitionsAdj).map(_.toLong))

  override def lower(
    ctx: ExecuteContext,
    requestedType: MatrixType,
    dropCols: Boolean,
    dropRows: Boolean,
  ): TableIR = {
    val nRowsAdj = if (dropRows) 0 else params.nRows
    val nColsAdj = if (dropCols) 0 else params.nCols
    var ht =
      TableRange(nRowsAdj, params.nPartitions.getOrElse(ctx.backend.defaultParallelism))
        .rename(Map("idx" -> "row_idx"))
    if (requestedType.colType.hasField(colUIDFieldName))
      ht = ht.mapGlobals(makeStruct(LowerMatrixIR.colsField ->
        irRange(0, nColsAdj).map('i ~> makeStruct(
          'col_idx -> 'i,
          Symbol(colUIDFieldName) -> 'i.toL,
        ))))
    else
      ht = ht.mapGlobals(makeStruct(LowerMatrixIR.colsField ->
        irRange(0, nColsAdj).map('i ~> makeStruct('col_idx -> 'i))))
    if (requestedType.rowType.hasField(rowUIDFieldName))
      ht = ht.mapRows('row.insertFields(
        LowerMatrixIR.entriesField -> irRange(0, nColsAdj).map('i ~> makeStruct()),
        Symbol(rowUIDFieldName) -> 'row('row_idx).toL,
      ))
    else
      ht = ht.mapRows('row.insertFields(
        LowerMatrixIR.entriesField ->
          irRange(0, nColsAdj).map('i ~> makeStruct())
      ))

    ht
  }

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixRangeReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixRangeReader => params == that.params
    case _ => false
  }
}

object MatrixRead {
  def apply(
    typ: MatrixType,
    dropCols: Boolean,
    dropRows: Boolean,
    reader: MatrixReader,
  ): MatrixRead = {
    assert(!reader.fullMatrixTypeWithoutUIDs.rowType.hasField(MatrixReader.rowUIDFieldName) &&
      !reader.fullMatrixTypeWithoutUIDs.colType.hasField(MatrixReader.colUIDFieldName))
    new MatrixRead(typ, dropCols, dropRows, reader)
  }

  def preserveExistingUIDs(
    typ: MatrixType,
    dropCols: Boolean,
    dropRows: Boolean,
    reader: MatrixReader,
  ): MatrixRead =
    new MatrixRead(typ, dropCols, dropRows, reader)
}

case class MatrixRead(
  typ: MatrixType,
  dropCols: Boolean,
  dropRows: Boolean,
  reader: MatrixReader,
) extends MatrixIR {
  assert(
    PruneDeadFields.isSupertype(typ, reader.fullMatrixType),
    s"\n  original:  ${reader.fullMatrixType}\n  requested: $typ",
  )

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    MatrixRead(typ, dropCols, dropRows, reader)
  }

  override def toString: String = s"MatrixRead($typ, " +
    s"partitionCounts = ${PartitionCounts(this)}, " +
    s"columnCount = ${ColumnCount(this)}, " +
    s"dropCols = $dropCols, " +
    s"dropRows = $dropRows)"

  final def lower(ctx: ExecuteContext): TableIR =
    reader.lower(ctx, typ, dropCols, dropRows)
}

case class MatrixFilterCols(child: MatrixIR, pred: IR) extends MatrixIR with PreservesRows {

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, pred)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixFilterCols = {
    assert(newChildren.length == 2)
    MatrixFilterCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  override def typ: MatrixType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixFilterRows(child: MatrixIR, pred: IR)
    extends MatrixIR with PreservesOrRemovesRows with PreservesCols {

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, pred)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixFilterRows = {
    assert(newChildren.length == 2)
    MatrixFilterRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  override def typ: MatrixType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixChooseCols(child: MatrixIR, oldIndices: IndexedSeq[Int])
    extends MatrixIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixChooseCols = {
    assert(newChildren.length == 1)
    MatrixChooseCols(newChildren(0).asInstanceOf[MatrixIR], oldIndices)
  }

  override def typ: MatrixType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixCollectColsByKey(child: MatrixIR) extends MatrixIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixCollectColsByKey = {
    assert(newChildren.length == 1)
    MatrixCollectColsByKey(newChildren(0).asInstanceOf[MatrixIR])
  }

  lazy val typ: MatrixType = {
    val newColValueType =
      TStruct(child.typ.colValueStruct.fields.map(f => f.copy(typ = TArray(f.typ))))
    val newColType = child.typ.colKeyStruct ++ newColValueType
    val newEntryType = TStruct(child.typ.entryType.fields.map(f => f.copy(typ = TArray(f.typ))))

    child.typ.copy(colType = newColType, entryType = newEntryType)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixAggregateRowsByKey(child: MatrixIR, entryExpr: IR, rowExpr: IR)
    extends MatrixIR with PreservesOrRemovesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, entryExpr, rowExpr)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixAggregateRowsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newRowExpr: IR) = newChildren
    MatrixAggregateRowsByKey(newChild, newEntryExpr, newRowExpr)
  }

  lazy val typ: MatrixType = child.typ.copy(
    rowType = child.typ.rowKeyStruct ++ tcoerce[TStruct](rowExpr.typ),
    entryType = tcoerce[TStruct](entryExpr.typ),
  )

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixAggregateColsByKey(child: MatrixIR, entryExpr: IR, colExpr: IR)
    extends MatrixIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, entryExpr, colExpr)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixAggregateColsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newColExpr: IR) = newChildren
    MatrixAggregateColsByKey(newChild, newEntryExpr, newColExpr)
  }

  lazy val typ = child.typ.copy(
    entryType = tcoerce[TStruct](entryExpr.typ),
    colType = child.typ.colKeyStruct ++ tcoerce[TStruct](colExpr.typ),
  )

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixUnionCols(left: MatrixIR, right: MatrixIR, joinType: String) extends MatrixIR {
  require(joinType == "inner" || joinType == "outer")

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixUnionCols = {
    assert(newChildren.length == 2)
    MatrixUnionCols(
      newChildren(0).asInstanceOf[MatrixIR],
      newChildren(1).asInstanceOf[MatrixIR],
      joinType,
    )
  }

  private def newRowType = {
    val leftKeyType = left.typ.rowKeyStruct
    val leftValueType = left.typ.rowValueStruct
    val rightValueType = right.typ.rowValueStruct
    if (
      leftValueType.fieldNames.toSet
        .intersect(rightValueType.fieldNames.toSet)
        .nonEmpty
    )
      throw new RuntimeException(
        s"invalid MatrixUnionCols: \n  left value:  $leftValueType\n  right value: $rightValueType"
      )

    leftKeyType ++ leftValueType ++ rightValueType
  }

  lazy val typ: MatrixType = if (joinType == "inner")
    left.typ.copy(rowType = newRowType)
  else
    left.typ.copy(
      rowType = newRowType,
      colType = TStruct(left.typ.colType.fields.map(f => f.copy(typ = f.typ))),
      entryType = TStruct(left.typ.entryType.fields.map(f => f.copy(typ = f.typ))),
    )
}

case class MatrixMapEntries(child: MatrixIR, newEntries: IR)
    extends MatrixIR with PreservesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, newEntries)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixMapEntries = {
    assert(newChildren.length == 2)
    MatrixMapEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  lazy val typ: MatrixType =
    child.typ.copy(entryType = tcoerce[TStruct](newEntries.typ))

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixKeyRowsBy(child: MatrixIR, keys: IndexedSeq[String], isSorted: Boolean = false)
    extends MatrixIR with PreservesRows with PreservesCols {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val typ: MatrixType = child.typ.copy(rowKey = keys)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixKeyRowsBy = {
    assert(newChildren.length == 1)
    MatrixKeyRowsBy(newChildren(0).asInstanceOf[MatrixIR], keys, isSorted)
  }

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

case class MatrixMapRows(child: MatrixIR, newRow: IR)
    extends MatrixIR with PreservesRows with PreservesCols {

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, newRow)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixMapRows = {
    assert(newChildren.length == 2)
    MatrixMapRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  lazy val typ: MatrixType =
    child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixMapCols(child: MatrixIR, newCol: IR, newKey: Option[IndexedSeq[String]])
    extends MatrixIR with PreservesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, newCol)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixMapCols = {
    assert(newChildren.length == 2)
    MatrixMapCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], newKey)
  }

  lazy val typ: MatrixType = {
    val newColType = newCol.typ.asInstanceOf[TStruct]
    val newColKey = newKey.getOrElse(child.typ.colKey)
    child.typ.copy(colKey = newColKey, colType = newColType)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixMapGlobals(child: MatrixIR, newGlobals: IR)
    extends MatrixIR with PreservesRows with PreservesCols {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, newGlobals)

  lazy val typ: MatrixType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixMapGlobals = {
    assert(newChildren.length == 2)
    MatrixMapGlobals(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixFilterEntries(child: MatrixIR, pred: IR)
    extends MatrixIR with PreservesRows with PreservesCols {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, pred)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixFilterEntries = {
    assert(newChildren.length == 2)
    MatrixFilterEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  override def typ: MatrixType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixAnnotateColsTable(
  child: MatrixIR,
  table: TableIR,
  root: String,
) extends MatrixIR with PreservesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child, table)

  lazy val typ: MatrixType = child.typ.copy(
    colType = child.typ.colType.structInsert(table.typ.valueType, FastSeq(root))
  )

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixAnnotateColsTable =
    MatrixAnnotateColsTable(
      newChildren(0).asInstanceOf[MatrixIR],
      newChildren(1).asInstanceOf[TableIR],
      root,
    )

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixAnnotateRowsTable(
  child: MatrixIR,
  table: TableIR,
  root: String,
  product: Boolean,
) extends MatrixIR with PreservesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child, table)

  private def annotationType =
    if (product)
      TArray(table.typ.valueType)
    else
      table.typ.valueType

  lazy val typ: MatrixType =
    child.typ.copy(rowType = child.typ.rowType.appendKey(root, annotationType))

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixAnnotateRowsTable = {
    val IndexedSeq(child: MatrixIR, table: TableIR) = newChildren
    MatrixAnnotateRowsTable(child, table, root, product)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixExplodeRows(child: MatrixIR, path: IndexedSeq[String])
    extends MatrixIR with PreservesCols {
  assert(path.nonEmpty)

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixExplodeRows = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeRows(newChild.asInstanceOf[MatrixIR], path)
  }

  lazy val typ: MatrixType = {
    val rowType = child.typ.rowType
    val f = rowType.fieldOption(path).getOrElse {
      throw new AssertionError(
        s"No such row field at path '${path.mkString("/")}' in matrix row type '$rowType'."
      )
    }
    child.typ.copy(rowType = rowType.structInsert(TIterable.elementType(f.typ), path))
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixRepartition(child: MatrixIR, n: Int, strategy: Int)
    extends MatrixIR with PreservesRows with PreservesCols {
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRepartition = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRepartition(newChild, n, strategy)
  }

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

case class MatrixUnionRows(childrenSeq: IndexedSeq[MatrixIR]) extends MatrixIR {
  require(childrenSeq.length > 1)

  override def typ: MatrixType = childrenSeq.head.typ

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixUnionRows =
    MatrixUnionRows(newChildren.asInstanceOf[IndexedSeq[MatrixIR]])
}

case class MatrixDistinctByRow(child: MatrixIR)
    extends MatrixIR with PreservesOrRemovesRows with PreservesCols {
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixDistinctByRow = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixDistinctByRow(newChild)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixRowsHead(child: MatrixIR, n: Long) extends MatrixIR with PreservesCols {
  require(n >= 0)
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRowsHead = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRowsHead(newChild, n)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixColsHead(child: MatrixIR, n: Int) extends MatrixIR with PreservesRows {
  require(n >= 0)
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixColsHead = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixColsHead(newChild, n)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixRowsTail(child: MatrixIR, n: Long) extends MatrixIR with PreservesCols {
  require(n >= 0)
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRowsTail = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRowsTail(newChild, n)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixColsTail(child: MatrixIR, n: Int) extends MatrixIR with PreservesRows {
  require(n >= 0)
  override def typ: MatrixType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixColsTail = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixColsTail(newChild, n)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixExplodeCols(child: MatrixIR, path: IndexedSeq[String])
    extends MatrixIR with PreservesRows {

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixExplodeCols = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeCols(newChild.asInstanceOf[MatrixIR], path)
  }

  lazy val typ: MatrixType = {
    val colType = child.typ.colType
    val f = colType.fieldOption(path).getOrElse {
      throw new AssertionError(
        s"No such column field at path '${path.mkString("/")}' in matrix row type '$colType'."
      )
    }

    child.typ.copy(colType = colType.structInsert(TIterable.elementType(f.typ), path))
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

/** Create a MatrixTable from a Table, where the column values are stored in a global field
  * 'colsFieldName', and the entry values are stored in a row field 'entriesFieldName'.
  */
case class CastTableToMatrix(
  child: TableIR,
  entriesFieldName: String,
  colsFieldName: String,
  colKey: IndexedSeq[String],
) extends MatrixIR with PreservesRows {
  lazy val typ: MatrixType =
    MatrixType.fromTableType(child.typ, colsFieldName, entriesFieldName, colKey)

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): CastTableToMatrix = {
    assert(newChildren.length == 1)
    CastTableToMatrix(
      newChildren(0).asInstanceOf[TableIR],
      entriesFieldName,
      colsFieldName,
      colKey,
    )
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixToMatrixApply(child: MatrixIR, function: MatrixToMatrixFunction)
    extends MatrixIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToMatrixApply(newChild, function)
  }

  override lazy val typ: MatrixType = function.typ(child.typ)

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesRowsCond: Boolean = function.preservesPartitionCounts
}

case class MatrixRename(
  child: MatrixIR,
  globalMap: Map[String, String],
  colMap: Map[String, String],
  rowMap: Map[String, String],
  entryMap: Map[String, String],
) extends MatrixIR with PreservesRows with PreservesCols {
  lazy val typ: MatrixType = MatrixType(
    globalType = child.typ.globalType.rename(globalMap),
    colKey = child.typ.colKey.map(k => colMap.getOrElse(k, k)),
    colType = child.typ.colType.rename(colMap),
    rowKey = child.typ.rowKey.map(k => rowMap.getOrElse(k, k)),
    rowType = child.typ.rowType.rename(rowMap),
    entryType = child.typ.entryType.rename(entryMap),
  )

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRename = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRename(newChild, globalMap, colMap, rowMap, entryMap)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixFilterIntervals(child: MatrixIR, intervals: IndexedSeq[Interval], keep: Boolean)
    extends MatrixIR with PreservesOrRemovesRows with PreservesCols {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixFilterIntervals(newChild, intervals, keep)
  }

  override def typ: MatrixType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class RelationalLetMatrixTable(name: Name, value: IR, body: MatrixIR)
    extends MatrixIR with PreservesRows {
  override def typ: MatrixType = body.typ

  override def childrenSeq: IndexedSeq[BaseIR] = Array(value, body)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newValue: IR, newBody: MatrixIR) = newChildren
    RelationalLetMatrixTable(name, newValue, newBody)
  }

  override def preservesRowsOrColsFrom: BaseIR = body
}
