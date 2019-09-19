
package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, RelationalFunctions}
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PInt32, PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.io.TextMatrixReader
import is.hail.io.bgen.MatrixBGENReader
import is.hail.io.gen.MatrixGENReader
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.table.AbstractTableSpec
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods

import scala.collection.mutable

object MatrixIR {
  def read(hc: HailContext, path: String, dropCols: Boolean = false, dropRows: Boolean = false, requestedType: Option[MatrixType]): MatrixIR = {
    val reader = MatrixNativeReader(path)
    MatrixRead(requestedType.getOrElse(reader.fullMatrixType), dropCols, dropRows, reader)
  }

  def range(hc: HailContext, nRows: Int, nCols: Int, nPartitions: Option[Int], dropCols: Boolean = false, dropRows: Boolean = false): MatrixIR = {
    val reader = MatrixRangeReader(nRows, nCols, nPartitions)
    MatrixRead(reader.fullMatrixType, dropCols = dropCols, dropRows = dropRows, reader = reader)
  }
}

abstract sealed class MatrixIR extends BaseIR {
  def typ: MatrixType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def columnCount: Option[Int] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR

  def persist(storageLevel: StorageLevel): MatrixIR = {
    ExecuteContext.scoped { ctx =>
      val tv = Interpret(this, ctx, optimize = true)
      MatrixLiteral(this.typ, TableLiteral(tv, ctx))
    }
  }

  def unpersist(): MatrixIR = {
    this match {
      case MatrixLiteral(typ, tl) => MatrixLiteral(typ, tl.unpersist().asInstanceOf[TableLiteral])
      case x => x
    }
  }

  def pyPersist(storageLevel: String): MatrixIR = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel: $storageLevel")
    }
    persist(level)
  }

  def pyUnpersist(): MatrixIR = unpersist()
}

object MatrixLiteral {
  def apply(typ: MatrixType, rvd: RVD, globals: Row, colValues: IndexedSeq[Row]): MatrixLiteral = {
    val tt = typ.canonicalTableType
    ExecuteContext.scoped { ctx =>
      MatrixLiteral(typ,
        TableLiteral(
          TableValue(tt,
            BroadcastRow(ctx, Row.merge(globals, Row(colValues)), typ.canonicalTableType.globalType),
            rvd),
          ctx))
    }
  }
}

case class MatrixLiteral(typ: MatrixType, tl: TableLiteral) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    MatrixLiteral(typ, tl)
  }

  override def toString: String = "MatrixLiteral(...)"
}

object MatrixReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[MatrixNativeReader],
      classOf[MatrixRangeReader],
      classOf[MatrixVCFReader],
      classOf[MatrixBGENReader],
      classOf[MatrixPLINKReader],
      classOf[MatrixGENReader],
      classOf[TextInputFilterAndReplace],
      classOf[TextMatrixReader])
  ) + new NativeReaderOptionsSerializer()
}

trait MatrixReader {
  def columnCount: Option[Int]

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullMatrixType: MatrixType

  def lower(mr: MatrixRead): TableIR
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

  def makeGlobalValue(ctx: ExecuteContext, requestedType: TableType, values: => IndexedSeq[Row]): BroadcastRow = {
    assert(fullType.globalType.size == 1)
    val colType = requestedType.globalType.fieldOption(LowerMatrixIR.colsFieldName)
      .map(fd => fd.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

    colType match {
      case Some(ct) =>
        assert(requestedType.globalType.size == 1)
        val containedFields = ct.fieldNames.toSet
        val colValueIndices = fullMatrixType.colType.fields
          .filter(f => containedFields.contains(f.name))
          .map(_.index)
          .toArray
        val arr = values.map(r => Row.fromSeq(colValueIndices.map(r.get))).toFastIndexedSeq
        BroadcastRow(ctx, Row(arr), requestedType.globalType)
      case None =>
        assert(requestedType.globalType == TStruct())
        BroadcastRow(ctx, Row(), requestedType.globalType)
    }
  }
}

case class MatrixNativeReader(
  path: String,
  options: Option[NativeReaderOptions] = None,
  _spec: AbstractMatrixTableSpec = null
) extends MatrixReader {
  lazy val spec: AbstractMatrixTableSpec = Option(_spec).getOrElse(
    (RelationalSpec.read(HailContext.get, path): @unchecked) match {
      case mts: AbstractMatrixTableSpec => mts
      case _: AbstractTableSpec => fatal(s"file is a Table, not a MatrixTable: '$path'")
    })

  lazy val columnCount: Option[Int] = Some(RelationalSpec.read(HailContext.get, path + "/cols")
    .asInstanceOf[AbstractTableSpec]
    .partitionCounts
    .sum
    .toInt)

  def partitionCounts: Option[IndexedSeq[Long]] = if (intervals.isEmpty) Some(spec.partitionCounts) else None

  def fullMatrixType: MatrixType = spec.matrix_type

  private def intervals = options.map(_.intervals)

  if (intervals.nonEmpty && !spec.indexed(path))
    fatal("""`intervals` specified on an unindexed matrix table.
            |This matrix table was written using an older version of hail
            |rewrite the matrix in order to create an index to proceed""".stripMargin)

  override def lower(mr: MatrixRead): TableIR = {
    val rowsPath = path + "/rows"
    val entriesPath = path + "/entries"
    val colsPath = path + "/cols"

    if (mr.dropCols) {
      val tt = TableType(mr.typ.rowType, mr.typ.rowKey, mr.typ.globalType)
      val trdr: TableReader = TableNativeReader(rowsPath, options, _spec = spec.rowsTableSpec(rowsPath))
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
        options,
        spec.rowsTableSpec(rowsPath),
        spec.entriesTableSpec(entriesPath))
      var tr: TableIR = TableRead(tt, mr.dropRows, trdr)
      val colsTable = TableRead(
        TableType(
          mr.typ.colType,
          FastIndexedSeq(),
          TStruct()
        ),
        dropRows = false,
        TableNativeReader(colsPath, _spec = spec.colsTableSpec(colsPath))
      )

      TableMapGlobals(tr, InsertFields(
        Ref("global", tr.typ.globalType),
        FastSeq(LowerMatrixIR.colsFieldName -> GetField(TableCollect(colsTable), "rows"))
      ))
    }
  }
}

case class MatrixRangeReader(nRows: Int, nCols: Int, nPartitions: Option[Int]) extends MatrixReader {
  val fullMatrixType: MatrixType = MatrixType(
    globalType = TStruct.empty(),
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32()),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32()),
    entryType = TStruct.empty())

  val columnCount: Option[Int] = Some(nCols)

  lazy val partitionCounts: Option[IndexedSeq[Long]] = {
    val nPartitionsAdj = math.min(nRows, nPartitions.getOrElse(HailContext.get.sc.defaultParallelism))
    Some(partition(nRows, nPartitionsAdj).map(_.toLong))
  }

  override def lower(mr: MatrixRead): TableIR = {
    val uid1 = Symbol(genUID())

    val nRowsAdj = if (mr.dropRows) 0 else nRows
    val nColsAdj = if (mr.dropCols) 0 else nCols
    TableRange(nRowsAdj, nPartitions.getOrElse(HailContext.get.sc.defaultParallelism))
      .rename(Map("idx" -> "row_idx"))
      .mapGlobals(makeStruct(LowerMatrixIR.colsField ->
        irRange(0, nColsAdj).map('i ~> makeStruct('col_idx -> 'i))))
      .mapRows('row.insertFields(LowerMatrixIR.entriesField ->
        irRange(0, nColsAdj).map('i ~> makeStruct())))
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
}

case class MatrixFilterRows(child: MatrixIR, pred: IR) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterRows = {
    assert(newChildren.length == 2)
    MatrixFilterRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  def typ: MatrixType = child.typ

  override def columnCount: Option[Int] = child.columnCount
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
}

case class MatrixUnionCols(left: MatrixIR, right: MatrixIR) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixUnionCols = {
    assert(newChildren.length == 2)
    MatrixUnionCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[MatrixIR])
  }

  val typ: MatrixType = left.typ

  override def columnCount: Option[Int] =
    left.columnCount.flatMap(leftCount => right.columnCount.map(rightCount => leftCount + rightCount))
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

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixExplodeRows = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeRows(newChild.asInstanceOf[MatrixIR], path)
  }

  override def columnCount: Option[Int] = child.columnCount

  val idx = Ref(genUID(), TInt32())

  val newRow: InsertFields = {
    val refs = path.init.scanLeft(Ref("va", child.typ.rowType))((struct, name) =>
      Ref(genUID(), coerce[TStruct](struct.typ).field(name).typ))

    path.zip(refs).zipWithIndex.foldRight[IR](idx) {
      case (((field, ref), i), arg) =>
        InsertFields(ref, FastIndexedSeq(field ->
          (if (i == refs.length - 1)
            ArrayRef(ToArray(GetField(ref, field)), arg)
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
}

case class MatrixUnionRows(children: IndexedSeq[MatrixIR]) extends MatrixIR {
  require(children.length > 1)
  require(children.map(_.typ).toSet.size == 1, children.map(_.typ))
  val typ: MatrixType = children.head.typ

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixUnionRows =
    MatrixUnionRows(newChildren.asInstanceOf[IndexedSeq[MatrixIR]])

  override def columnCount: Option[Int] =
    children.map(_.columnCount).reduce { (c1, c2) =>
      require(c1.forall { i1 => c2.forall(i1 == _) })
      c1.orElse(c2)
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
}

case class MatrixRowsHead(child: MatrixIR, n: Long) extends MatrixIR {
  require(n >= 0)
  val typ: MatrixType = child.typ

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsHead = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixRowsHead(newChild, n)
  }

  override def columnCount: Option[Int] = child.columnCount
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
}

case class MatrixExplodeCols(child: MatrixIR, path: IndexedSeq[String]) extends MatrixIR {

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixExplodeCols = {
    val IndexedSeq(newChild) = newChildren
    MatrixExplodeCols(newChild.asInstanceOf[MatrixIR], path)
  }

  override def columnCount: Option[Int] = None

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  private val (keysType, querier) = child.typ.colType.queryTyped(path.toList)
  private val keyType = keysType match {
    case TArray(e, _) => e
    case TSet(e, _) => e
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
    case TArray(TStruct(_, _), _) =>
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
}

case class RelationalLetMatrixTable(name: String, value: IR, body: MatrixIR) extends MatrixIR {
  def typ: MatrixType = body.typ

  def children: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newValue: IR, newBody: MatrixIR) = newChildren
    RelationalLetMatrixTable(name, newValue, newBody)
  }
}
