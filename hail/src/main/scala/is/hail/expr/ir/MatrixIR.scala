
package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.Backend
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.ir.functions.MatrixToMatrixFunction
import is.hail.types._
import is.hail.types.virtual._
import is.hail.io.TextMatrixReader
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.fs.FS
import is.hail.io.gen.MatrixGENReader
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.rvd._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s._

object MatrixIR {
  def read(fs: FS, path: String, dropCols: Boolean = false, dropRows: Boolean = false, requestedType: Option[MatrixType] = None): MatrixIR = {
    val reader = MatrixNativeReader(fs, path)
    MatrixRead(requestedType.getOrElse(reader.fullMatrixType), dropCols, dropRows, reader)
  }

  def range(nRows: Int, nCols: Int, nPartitions: Option[Int], dropCols: Boolean = false, dropRows: Boolean = false): MatrixIR = {
    val reader = MatrixRangeReader(nRows, nCols, nPartitions)
    MatrixRead(reader.fullMatrixType, dropCols = dropCols, dropRows = dropRows, reader = reader)
  }
}

abstract sealed class MatrixIR extends BaseIR {
  def typ: MatrixType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  val rowCountUpperBound: Option[Long]

  def columnCount: Option[Int] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR

  def unpersist(): MatrixIR = {
    this match {
      case MatrixLiteral(typ, tl) => MatrixLiteral(typ, tl.unpersist().asInstanceOf[TableLiteral])
      case x => x
    }
  }

  def pyUnpersist(): MatrixIR = unpersist()
}

object MatrixLiteral {
  def apply(ctx: ExecuteContext, typ: MatrixType, rvd: RVD, globals: Row, colValues: IndexedSeq[Row]): MatrixLiteral = {
    val tt = typ.canonicalTableType
    MatrixLiteral(typ,
      TableLiteral(
        TableValue(ctx, tt,
          BroadcastRow(ctx, Row.fromSeq(globals.toSeq :+ colValues), typ.canonicalTableType.globalType),
          rvd)))
  }
}

case class MatrixLiteral(typ: MatrixType, tl: TableLiteral) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    MatrixLiteral(typ, tl)
  }

  override def toString: String = "MatrixLiteral(...)"
}

object MatrixReader {
  def fromJson(env: IRParserEnvironment, jv: JValue): MatrixReader = {
    implicit val formats: Formats = DefaultFormats
    (jv \ "name").extract[String] match {
      case "MatrixRangeReader" => MatrixRangeReader.fromJValue(env.ctx, jv)
      case "MatrixNativeReader" => MatrixNativeReader.fromJValue(env.ctx.fs, jv)
      case "MatrixBGENReader" => MatrixBGENReader.fromJValue(env, jv)
      case "TextMatrixReader" => TextMatrixReader.fromJValue(env.ctx, jv)
      case "MatrixGENReader" => MatrixGENReader.fromJValue(env.ctx, jv)
      case "MatrixPLINKReader" => MatrixPLINKReader.fromJValue(env.ctx, jv)
      case "MatrixVCFReader" => MatrixVCFReader.fromJValue(env.ctx, jv)
    }
  }
}

trait MatrixReader {
  def pathsUsed: Seq[String]

  def columnCount: Option[Int]

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullMatrixType: MatrixType

  def lower(mr: MatrixRead): TableIR

  def toJValue: JValue
}

abstract class MatrixHybridReader extends TableReader with MatrixReader {
  lazy val fullType: TableType = fullMatrixType.canonicalTableType

  override def lower(mr: MatrixRead): TableIR = {
    var tr: TableIR = TableRead(mr.typ.canonicalTableType, mr.dropRows, this)
    if (mr.dropCols) {
      // this lowering preserves dropCols using pruning
      tr = TableMapRows(
        tr,
        InsertFields(
          Ref("row", tr.typ.rowType),
          FastIndexedSeq(LowerMatrixIR.entriesFieldName -> MakeArray(FastSeq(), TArray(mr.typ.entryType)))))
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref("global", tr.typ.globalType),
          FastIndexedSeq(LowerMatrixIR.colsFieldName -> MakeArray(FastSeq(), TArray(mr.typ.colType)))))
    }
    tr
  }

  def makeGlobalValue(ctx: ExecuteContext, requestedType: TStruct, values: => IndexedSeq[Row]): BroadcastRow = {
    assert(fullType.globalType.size == 1)
    val colType = requestedType.fieldOption(LowerMatrixIR.colsFieldName)
      .map(fd => fd.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

    colType match {
      case Some(ct) =>
        assert(requestedType.size == 1)
        val containedFields = ct.fieldNames.toSet
        val colValueIndices = fullMatrixType.colType.fields
          .filter(f => containedFields.contains(f.name))
          .map(_.index)
          .toArray
        val arr = values.map(r => Row.fromSeq(colValueIndices.map(r.get))).toFastIndexedSeq
        BroadcastRow(ctx, Row(arr), requestedType)
      case None =>
        assert(requestedType == TStruct.empty)
        BroadcastRow(ctx, Row(), requestedType)
    }
  }
}

object MatrixNativeReader {
  def apply(fs: FS, path: String, options: Option[NativeReaderOptions] = None): MatrixNativeReader =
    MatrixNativeReader(fs, MatrixNativeReaderParameters(path, options))

  def apply(fs: FS, params: MatrixNativeReaderParameters): MatrixNativeReader = {
    val spec =
      (RelationalSpec.read(fs, params.path): @unchecked) match {
        case mts: AbstractMatrixTableSpec => mts
        case _: AbstractTableSpec => fatal(s"file is a Table, not a MatrixTable: '${ params.path }'")
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
  options: Option[NativeReaderOptions])

class MatrixNativeReader(
  val params: MatrixNativeReaderParameters,
  spec: AbstractMatrixTableSpec
) extends MatrixReader {
  def pathsUsed: Seq[String] = FastSeq(params.path)

  lazy val columnCount: Option[Int] = Some(spec.colsSpec
    .partitionCounts
    .sum
    .toInt)

  def partitionCounts: Option[IndexedSeq[Long]] = if (params.options.isEmpty) Some(spec.partitionCounts) else None

  def fullMatrixType: MatrixType = spec.matrix_type

  override def lower(mr: MatrixRead): TableIR = {
    val rowsPath = params.path + "/rows"
    val entriesPath = params.path + "/entries"
    val colsPath = params.path + "/cols"

    if (mr.dropCols) {
      val tt = TableType(mr.typ.rowType, mr.typ.rowKey, mr.typ.globalType)
      val trdr: TableReader = new TableNativeReader(TableNativeReaderParameters(rowsPath, params.options), spec.rowsSpec)
      var tr: TableIR = TableRead(tt, mr.dropRows, trdr)
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref("global", tr.typ.globalType),
          FastSeq(LowerMatrixIR.colsFieldName -> MakeArray(FastSeq(), TArray(mr.typ.colType)))))
      TableMapRows(
        tr,
        InsertFields(
          Ref("row", tr.typ.rowType),
        FastSeq(LowerMatrixIR.entriesFieldName -> MakeArray(FastSeq(), TArray(mr.typ.entryType)))))
    } else {
      val tt = TableType(
        mr.typ.rowType.appendKey(LowerMatrixIR.entriesFieldName, TArray(mr.typ.entryType)),
        mr.typ.rowKey,
        mr.typ.globalType)
      val trdr = TableNativeZippedReader(
        rowsPath,
        entriesPath,
        params.options,
        spec.rowsSpec,
        spec.entriesSpec)
      val tr: TableIR = TableRead(tt, mr.dropRows, trdr)
      val colsRVDSpec = spec.colsSpec.rowsSpec
      val partFiles = colsRVDSpec.absolutePartPaths(spec.colsSpec.rowsComponent.absolutePath(colsPath))

      val cols = if (partFiles.length == 1) {
        ReadPartition(Str(partFiles.head), mr.typ.colType, PartitionNativeReader(colsRVDSpec.typedCodecSpec))
      } else {
        val partNames = MakeArray(partFiles.map(Str), TArray(TString))
        val elt = Ref(genUID(), TString)
        StreamFlatMap(
          partNames,
          elt.name,
          ReadPartition(elt, mr.typ.colType, PartitionNativeReader(colsRVDSpec.typedCodecSpec)))
      }

      TableMapGlobals(tr, InsertFields(
        Ref("global", tr.typ.globalType),
        FastSeq(LowerMatrixIR.colsFieldName -> ToArray(cols))
      ))
    }
  }

  def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixNativeReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixNativeReader => params == that.params
    case _ => false
  }
}

object MatrixRangeReader {
  def apply(nRows: Int, nCols: Int, nPartitions: Option[Int]): MatrixRangeReader =
    MatrixRangeReader(MatrixRangeReaderParameters(nRows, nCols, nPartitions))

  def fromJValue(ctx: ExecuteContext, jv: JValue): MatrixRangeReader = {
    implicit val formats: Formats = DefaultFormats
    val params = jv.extract[MatrixRangeReaderParameters]

    MatrixRangeReader(params)
  }

  def apply(params: MatrixRangeReaderParameters): MatrixRangeReader = {
    val nPartitionsAdj = math.min(params.nRows, params.nPartitions.getOrElse(HailContext.backend.defaultParallelism))
    new MatrixRangeReader(params, nPartitionsAdj)
  }
}

case class MatrixRangeReaderParameters(nRows: Int, nCols: Int, nPartitions: Option[Int])

class MatrixRangeReader(
  val params: MatrixRangeReaderParameters,
  nPartitionsAdj: Int
) extends MatrixReader {
  def pathsUsed: Seq[String] = FastSeq()
  val fullMatrixType: MatrixType = MatrixType(
    globalType = TStruct.empty,
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32),
    entryType = TStruct.empty)

  val columnCount: Option[Int] = Some(params.nCols)

  lazy val partitionCounts: Option[IndexedSeq[Long]] = Some(partition(params.nRows, nPartitionsAdj).map(_.toLong))

  override def lower(mr: MatrixRead): TableIR = {
    val nRowsAdj = if (mr.dropRows) 0 else params.nRows
    val nColsAdj = if (mr.dropCols) 0 else params.nCols
    TableRange(nRowsAdj, params.nPartitions.getOrElse(HailContext.backend.defaultParallelism))
      .rename(Map("idx" -> "row_idx"))
      .mapGlobals(makeStruct(LowerMatrixIR.colsField ->
        irRange(0, nColsAdj).map('i ~> makeStruct('col_idx -> 'i))))
      .mapRows('row.insertFields(LowerMatrixIR.entriesField ->
        irRange(0, nColsAdj).map('i ~> makeStruct())))
  }

  def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "MatrixRangeReader")
  }

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixRangeReader => params == that.params
    case _ => false
  }
}

case class MatrixRead(
  typ: MatrixType,
  dropCols: Boolean,
  dropRows: Boolean,
  reader: MatrixReader) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    MatrixRead(typ, dropCols, dropRows, reader)
  }

  override def toString: String = s"MatrixRead($typ, " +
    s"partitionCounts = $partitionCounts, " +
    s"columnCount = $columnCount, " +
    s"dropCols = $dropCols, " +
    s"dropRows = $dropRows)"

  override def partitionCounts: Option[IndexedSeq[Long]] = {
    if (dropRows)
      Some(Array.empty[Long])
    else
      reader.partitionCounts
  }

  lazy val rowCountUpperBound: Option[Long] = partitionCounts.map(_.sum)

  override def columnCount: Option[Int] = {
    if (dropCols)
      Some(0)
    else
      reader.columnCount
  }

  final def lower(): TableIR = reader.lower(this)
}

case class MatrixFilterCols(child: MatrixIR, pred: IR) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterCols = {
    assert(newChildren.length == 2)
    MatrixFilterCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = child.typ

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixFilterRows(child: MatrixIR, pred: IR) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterRows = {
    assert(newChildren.length == 2)
    MatrixFilterRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  def typ: MatrixType = child.typ

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixChooseCols(child: MatrixIR, oldIndices: IndexedSeq[Int]) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixChooseCols = {
    assert(newChildren.length == 1)
    MatrixChooseCols(newChildren(0).asInstanceOf[MatrixIR], oldIndices)
  }

  val typ: MatrixType = child.typ

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = Some(oldIndices.length)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixCollectColsByKey(child: MatrixIR) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixCollectColsByKey = {
    assert(newChildren.length == 1)
    MatrixCollectColsByKey(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: MatrixType = {
    val newColValueType = TStruct(child.typ.colValueStruct.fields.map(f => f.copy(typ = TArray(f.typ))))
    val newColType = child.typ.colKeyStruct ++ newColValueType
    val newEntryType = TStruct(child.typ.entryType.fields.map(f => f.copy(typ = TArray(f.typ))))

    child.typ.copy(colType = newColType, entryType = newEntryType)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixAggregateRowsByKey(child: MatrixIR, entryExpr: IR, rowExpr: IR) extends MatrixIR {
  require(child.typ.rowKey.nonEmpty)

  lazy val children: IndexedSeq[BaseIR] = Array(child, entryExpr, rowExpr)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateRowsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newRowExpr: IR) = newChildren
    MatrixAggregateRowsByKey(newChild, newEntryExpr, newRowExpr)
  }

  val typ: MatrixType = child.typ.copy(
    rowType = child.typ.rowKeyStruct ++ coerce[TStruct](rowExpr.typ),
    entryType = coerce[TStruct](entryExpr.typ)
  )

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixAggregateColsByKey(child: MatrixIR, entryExpr: IR, colExpr: IR) extends MatrixIR {
  require(child.typ.colKey.nonEmpty)

  lazy val children: IndexedSeq[BaseIR] = Array(child, entryExpr, colExpr)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateColsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newColExpr: IR) = newChildren
    MatrixAggregateColsByKey(newChild, newEntryExpr, newColExpr)
  }

  val typ = child.typ.copy(
    entryType = coerce[TStruct](entryExpr.typ),
    colType = child.typ.colKeyStruct ++ coerce[TStruct](colExpr.typ))

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixUnionCols(left: MatrixIR, right: MatrixIR, joinType: String) extends MatrixIR {
  require(joinType == "inner" || joinType == "outer")
  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixUnionCols = {
    assert(newChildren.length == 2)
    MatrixUnionCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[MatrixIR], joinType)
  }

  val typ: MatrixType = if (joinType == "inner")
    left.typ
  else
    left.typ.copy(
      colType = TStruct(left.typ.colType.fields.map(f => f.copy(typ = f.typ))),
      entryType = TStruct(left.typ.entryType.fields.map(f => f.copy(typ = f.typ))))

  override def columnCount: Option[Int] =
    left.columnCount.flatMap(leftCount => right.columnCount.map(rightCount => leftCount + rightCount))

  lazy val rowCountUpperBound: Option[Long] = (left.rowCountUpperBound, right.rowCountUpperBound) match {
    case (Some(l), Some(r)) => if (joinType == "inner") Some(l.min(r)) else Some(l + r)
    case (Some(l), None) => if (joinType == "inner") Some(l) else None
    case (None, Some(r)) => if (joinType == "inner") Some(r) else None
    case (None, None) => None
  }
}

case class MatrixMapEntries(child: MatrixIR, newEntries: IR) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child, newEntries)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapEntries = {
    assert(newChildren.length == 2)
    MatrixMapEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType =
    child.typ.copy(entryType = coerce[TStruct](newEntries.typ))

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixKeyRowsBy(child: MatrixIR, keys: IndexedSeq[String], isSorted: Boolean = false) extends MatrixIR {
  private val fields = child.typ.rowType.fieldNames.toSet
  assert(keys.forall(fields.contains), s"${ keys.filter(k => !fields.contains(k)).mkString(", ") }")

  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: MatrixType = child.typ.copy(rowKey = keys)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixKeyRowsBy = {
    assert(newChildren.length == 1)
    MatrixKeyRowsBy(newChildren(0).asInstanceOf[MatrixIR], keys, isSorted)
  }

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixMapRows(child: MatrixIR, newRow: IR) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = Array(child, newRow)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapRows = {
    assert(newChildren.length == 2)
    MatrixMapRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = {
    child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixMapCols(child: MatrixIR, newCol: IR, newKey: Option[IndexedSeq[String]]) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child, newCol)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapCols = {
    assert(newChildren.length == 2)
    MatrixMapCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], newKey)
  }

  val typ: MatrixType = {
    val newColType = newCol.typ.asInstanceOf[TStruct]
    val newColKey = newKey.getOrElse(child.typ.colKey)
    child.typ.copy(colKey = newColKey, colType = newColType)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixMapGlobals(child: MatrixIR, newGlobals: IR) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, newGlobals)

  val typ: MatrixType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapGlobals = {
    assert(newChildren.length == 2)
    MatrixMapGlobals(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixFilterEntries(child: MatrixIR, pred: IR) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterEntries = {
    assert(newChildren.length == 2)
    MatrixFilterEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = child.typ

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixAnnotateColsTable(
  child: MatrixIR,
  table: TableIR,
  root: String) extends MatrixIR {
  require(child.typ.colType.fieldOption(root).isEmpty)

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child, table)

  override def columnCount: Option[Call] = child.columnCount

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  private val (colType, inserter) = child.typ.colType.structInsert(table.typ.valueType, List(root))
  val typ: MatrixType = child.typ.copy(colType = colType)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAnnotateColsTable = {
    MatrixAnnotateColsTable(
      newChildren(0).asInstanceOf[MatrixIR],
      newChildren(1).asInstanceOf[TableIR],
      root)
  }

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixAnnotateRowsTable(
  child: MatrixIR,
  table: TableIR,
  root: String,
  product: Boolean
) extends MatrixIR {
  require((!product && table.typ.keyType.isPrefixOf(child.typ.rowKeyStruct)) ||
    (table.typ.keyType.size == 1 && table.typ.keyType.types(0) == TInterval(child.typ.rowKeyStruct.types(0))),
    s"\n  L: ${ child.typ }\n  R: ${ table.typ }")

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child, table)

  override def columnCount: Option[Int] = child.columnCount

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  private val annotationType =
    if (product)
      TArray(table.typ.valueType)
    else
      table.typ.valueType

  val typ: MatrixType =
    child.typ.copy(rowType = child.typ.rowType.appendKey(root, annotationType))

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAnnotateRowsTable = {
    val IndexedSeq(child: MatrixIR, table: TableIR) = newChildren
    MatrixAnnotateRowsTable(child, table, root, product)
  }
}

case class MatrixExplodeRows(child: MatrixIR, path: IndexedSeq[String]) extends MatrixIR {
  assert(path.nonEmpty)

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixExplodeRows = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeRows(newChild.asInstanceOf[MatrixIR], path)
  }

  override def columnCount: Option[Int] = child.columnCount

  val idx = Ref(genUID(), TInt32)

  val newRow: InsertFields = {
    val refs = path.init.scanLeft(Ref("va", child.typ.rowType))((struct, name) =>
      Ref(genUID(), coerce[TStruct](struct.typ).field(name).typ))

    path.zip(refs).zipWithIndex.foldRight[IR](idx) {
      case (((field, ref), i), arg) =>
        InsertFields(ref, FastIndexedSeq(field ->
          (if (i == refs.length - 1)
            ArrayRef(ToArray(ToStream(GetField(ref, field))), arg)
          else
            Let(refs(i + 1).name, GetField(ref, field), arg))))
    }.asInstanceOf[InsertFields]
  }

  val typ: MatrixType = child.typ.copy(rowType = newRow.typ)
}

case class MatrixRepartition(child: MatrixIR, n: Int, strategy: Int) extends MatrixIR {
  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRepartition = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRepartition(newChild, n, strategy)
  }

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixUnionRows(children: IndexedSeq[MatrixIR]) extends MatrixIR {
  require(children.length > 1)
  require(children.tail.forall(c => compatible(c.typ, children.head.typ)), children.map(_.typ))
  val typ: MatrixType = children.head.typ

  def compatible(t1: MatrixType, t2: MatrixType): Boolean = {
    t1.colKeyStruct == t2.colKeyStruct &&
      t1.rowType == t2.rowType &&
      t1.rowKey == t2.rowKey &&
      t1.entryType == t2.entryType
  }

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixUnionRows =
    MatrixUnionRows(newChildren.asInstanceOf[IndexedSeq[MatrixIR]])

  override def columnCount: Option[Int] =
    children.map(_.columnCount).reduce { (c1, c2) =>
      require(c1.forall { i1 => c2.forall(i1 == _) })
      c1.orElse(c2)
    }

  lazy val rowCountUpperBound: Option[Long] = {
    val definedChildren = children.flatMap(_.rowCountUpperBound)
    if (definedChildren.length == children.length)
      Some(definedChildren.sum)
    else
      None
  }
}

case class MatrixDistinctByRow(child: MatrixIR) extends MatrixIR {

  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixDistinctByRow = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixDistinctByRow(newChild)
  }

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixRowsHead(child: MatrixIR, n: Long) extends MatrixIR {
  require(n >= 0)
  val typ: MatrixType = child.typ

  override lazy val partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts.map { pc =>
    val prefixSums = pc.iterator.scanLeft(0L)(_ + _)
    val newPCs = pc.iterator.zip(prefixSums)
      .takeWhile { case (_, prefixSum) => prefixSum < n }
      .map { case (value, prefixSum) => if (prefixSum + value > n) n - prefixSum else value }
      .toFastIndexedSeq
    assert(newPCs.sum == n || pc.sum < n)
    newPCs
  }

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsHead = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRowsHead(newChild, n)
  }

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound match {
    case Some(c) => Some(c.min(n))
    case None => Some(n)
  }
}

case class MatrixColsHead(child: MatrixIR, n: Int) extends MatrixIR {
  require(n >= 0)
  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsHead = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixColsHead(newChild, n)
  }

  override def columnCount: Option[Int] = child.columnCount.map(math.min(_, n))

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixRowsTail(child: MatrixIR, n: Long) extends MatrixIR {
  require(n >= 0)
  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsTail = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRowsTail(newChild, n)
  }

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound match {
    case Some(c) => Some(c.min(n))
    case None => Some(n)
  }
}

case class MatrixColsTail(child: MatrixIR, n: Int) extends MatrixIR {
  require(n >= 0)
  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTail = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixColsTail(newChild, n)
  }

  override def columnCount: Option[Int] = child.columnCount.map(math.min(_, n))

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixExplodeCols(child: MatrixIR, path: IndexedSeq[String]) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixExplodeCols = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeCols(newChild.asInstanceOf[MatrixIR], path)
  }

  override def columnCount: Option[Int] = None

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  private val (keysType, querier) = child.typ.colType.queryTyped(path.toList)
  private val keyType = keysType match {
    case TArray(e) => e
    case TSet(e) => e
  }
  val (newColType, inserter) = child.typ.colType.structInsert(keyType, path.toList)
  val typ: MatrixType = child.typ.copy(colType = newColType)
}

/** Create a MatrixTable from a Table, where the column values are stored in a
  * global field 'colsFieldName', and the entry values are stored in a row
  * field 'entriesFieldName'.
  */
case class CastTableToMatrix(
  child: TableIR,
  entriesFieldName: String,
  colsFieldName: String,
  colKey: IndexedSeq[String]
) extends MatrixIR {

  child.typ.rowType.fieldType(entriesFieldName) match {
    case TArray(TStruct(_)) =>
    case t => fatal(s"expected entry field to be an array of structs, found $t")
  }

  val typ: MatrixType = MatrixType.fromTableType(child.typ, colsFieldName, entriesFieldName, colKey)

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): CastTableToMatrix = {
    assert(newChildren.length == 1)
    CastTableToMatrix(
      newChildren(0).asInstanceOf[TableIR],
      entriesFieldName,
      colsFieldName,
      colKey)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class MatrixToMatrixApply(child: MatrixIR, function: MatrixToMatrixFunction) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToMatrixApply(newChild, function)
  }

  override lazy val typ: MatrixType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None

  lazy val rowCountUpperBound: Option[Long] = if (function.preservesPartitionCounts) child.rowCountUpperBound else None
}

case class MatrixRename(child: MatrixIR,
  globalMap: Map[String, String], colMap: Map[String, String], rowMap: Map[String, String], entryMap: Map[String, String]) extends MatrixIR {
  require(globalMap.keys.forall(child.typ.globalType.hasField))
  require(colMap.keys.forall(child.typ.colType.hasField))
  require(rowMap.keys.forall(child.typ.rowType.hasField))
  require(entryMap.keys.forall(child.typ.entryType.hasField))

  lazy val typ: MatrixType = MatrixType(
    globalType = child.typ.globalType.rename(globalMap),
    colKey = child.typ.colKey.map(k => colMap.getOrElse(k, k)),
    colType = child.typ.colType.rename(colMap),
    rowKey = child.typ.rowKey.map(k => rowMap.getOrElse(k, k)),
    rowType = child.typ.rowType.rename(rowMap),
    entryType = child.typ.entryType.rename(entryMap))

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  override def columnCount: Option[Int] = child.columnCount

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRename = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRename(newChild, globalMap, colMap, rowMap, entryMap)
  }
}

case class MatrixFilterIntervals(child: MatrixIR, intervals: IndexedSeq[Interval], keep: Boolean) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixFilterIntervals(newChild, intervals, keep)
  }

  override lazy val typ: MatrixType = child.typ

  override def columnCount: Option[Int] = child.columnCount

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound
}

case class RelationalLetMatrixTable(name: String, value: IR, body: MatrixIR) extends MatrixIR {
  def typ: MatrixType = body.typ

  def children: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newValue: IR, newBody: MatrixIR) = newChildren
    RelationalLetMatrixTable(name, newValue, newBody)
  }

  lazy val rowCountUpperBound: Option[Long] = body.rowCountUpperBound
}
