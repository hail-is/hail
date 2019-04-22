
package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.ir
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, RelationalFunctions}
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PInt32, PStruct, PType}
import is.hail.expr.types.virtual._
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

  def chooseColsWithArray(typ: MatrixType): (MatrixType, (MatrixValue, Array[Int]) => MatrixValue) = {
    val rowType = typ.rvRowType
    val keepType = TArray(+TInt32())
    val (rTyp, makeF) = ir.Compile[Long, Long, Long]("row", rowType.physicalType,
      "keep", keepType.physicalType,
      body = InsertFields(ir.Ref("row", rowType), Seq((MatrixType.entriesIdentifier,
        ir.ArrayMap(ir.Ref("keep", keepType), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), MatrixType.entriesIdentifier),
            ir.Ref("i", TInt32())))))))
    assert(rTyp.isOfType(rowType.physicalType))

    val newMatrixType = typ.copy(rvRowType = coerce[TStruct](rTyp.virtualType))

    val keepF = { (mv: MatrixValue, keep: Array[Int]) =>
      val keepBc = mv.sparkContext.broadcast(keep)
      mv.copy(typ = newMatrixType,
        colValues = mv.colValues.copy(value = keep.map(mv.colValues.value)),
        rvd = mv.rvd.mapPartitionsWithIndex(newMatrixType.canonicalRVDType, { (i, ctx, it) =>
          val f = makeF(i)
          val keep = keepBc.value
          val rv2 = RegionValue()

          it.map { rv =>
            val region = ctx.region
            rv2.set(region,
              f(region, rv.offset, false, region.appendArrayInt(keep), false))
            rv2
          }
        }))
    }
    (newMatrixType, keepF)
  }

  def filterCols(typ: MatrixType): (MatrixType, (MatrixValue, (Annotation, Int) => Boolean) => MatrixValue) = {
    val (t, keepF) = chooseColsWithArray(typ)
    (t, { (mv: MatrixValue, p: (Annotation, Int) => Boolean) =>
      val keep = (0 until mv.nCols)
        .view
        .filter { i => p(mv.colValues.value(i), i) }
        .toArray
      keepF(mv, keep)
    })
  }
}

abstract sealed class MatrixIR extends BaseIR {
  def typ: MatrixType

  def rvdType: RVDType = typ.canonicalRVDType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def getOrComputePartitionCounts(): IndexedSeq[Long] = {
    partitionCounts
      .getOrElse(
        Interpret(
          TableMapRows(
            TableKeyBy(MatrixRowsTable(this), FastIndexedSeq()),
            MakeStruct(FastIndexedSeq())
          ))
          .rvd
          .countPerPartition()
          .toFastIndexedSeq)
  }

  def columnCount: Option[Int] = None

  protected[ir] def execute(hc: HailContext): MatrixValue =
    fatal("tried to execute unexecutable IR")

  override def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR

  def persist(storageLevel: StorageLevel): MatrixIR = {
    val mv = Interpret(this)
    MatrixLiteral(mv.persist(storageLevel))
  }

  def unpersist(): MatrixIR = {
    val mv = Interpret(this)
    MatrixLiteral(mv.unpersist())
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

case class MatrixLiteral(value: MatrixValue) extends MatrixIR {
  val typ: MatrixType = value.typ

  override val rvdType: RVDType = value.rvd.typ

  lazy val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  protected[ir] override def execute(hc: HailContext): MatrixValue = value

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    MatrixLiteral(value)
  }

  override def columnCount: Option[Int] = Some(value.nCols)

  override def toString: String = "MatrixLiteral(...)"
}

object MatrixReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[MatrixNativeReader], classOf[MatrixRangeReader], classOf[MatrixVCFReader],
      classOf[MatrixBGENReader], classOf[MatrixPLINKReader], classOf[MatrixGENReader],
      classOf[TextInputFilterAndReplace]))
}

trait MatrixReader {
  def columnCount: Option[Int]

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullMatrixType: MatrixType

  // TODO: remove fullRVDType when lowering is finished
  def fullRVDType: RVDType

  def lower(mr: MatrixRead): TableIR
}

abstract class MatrixHybridReader extends TableReader with MatrixReader {
  lazy val fullType: TableType = LowerMatrixIR.loweredType(fullMatrixType)

  override def lower(mr: MatrixRead): TableIR = {
    var tr: TableIR = TableRead(LowerMatrixIR.loweredType(mr.typ), mr.dropRows, this)
    if (mr.dropCols) {
      // this lowering preserves dropCols using pruning
      tr = TableMapRows(
        tr,
        InsertFields(
          Ref("row", tr.typ.rowType),
          FastIndexedSeq(LowerMatrixIR.entriesFieldName -> MakeArray(FastSeq(), mr.typ.entryArrayType))))
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref("global", tr.typ.globalType),
          FastIndexedSeq(LowerMatrixIR.colsFieldName -> MakeArray(FastSeq(), TArray(mr.typ.colType)))))
    }
    tr
  }

  def makeGlobalValue(requestedType: TableType, values: => IndexedSeq[Row]): BroadcastRow = {
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
        BroadcastRow(Row(arr), requestedType.globalType, HailContext.get.sc)
      case None =>
        assert(requestedType.globalType == TStruct())
        BroadcastRow(Row.empty, requestedType.globalType, HailContext.get.sc)
    }
  }
}

case class MatrixNativeReader(path: String, _spec: AbstractMatrixTableSpec = null) extends MatrixReader {
  lazy val spec: AbstractMatrixTableSpec = Option(_spec).getOrElse(
    (RelationalSpec.read(HailContext.get, path): @unchecked) match {
      case mts: AbstractMatrixTableSpec => mts
      case _: AbstractTableSpec => fatal(s"file is a Table, not a MatrixTable: '$path'")
    })

  override lazy val fullRVDType: RVDType = spec.rvdType(path)

  lazy val columnCount: Option[Int] = Some(RelationalSpec.read(HailContext.get, path + "/cols")
    .asInstanceOf[AbstractTableSpec]
    .partitionCounts
    .sum
    .toInt)

  def partitionCounts: Option[IndexedSeq[Long]] = Some(spec.partitionCounts)

  def fullMatrixType: MatrixType = spec.matrix_type

  override def lower(mr: MatrixRead): TableIR = {
    val rowsPath = path + "/rows"
    val entriesPath = path + "/entries"
    val colsPath = path + "/cols"

    val hc = HailContext.get

    var tr: TableIR = TableRead(
      TableType(
        mr.typ.rowType,
        mr.typ.rowKey,
        mr.typ.globalType
      ),
      mr.dropRows,
      TableNativeReader(rowsPath, spec.rowsTableSpec(rowsPath))
    )

    if (mr.dropCols) {
      tr = TableMapGlobals(
        tr,
        InsertFields(
          Ref("global", tr.typ.globalType),
          FastSeq(LowerMatrixIR.colsFieldName -> MakeArray(FastSeq(), TArray(mr.typ.colType)))))
      tr = TableMapRows(
        tr,
        InsertFields(
          Ref("row", tr.typ.rowType),
        FastSeq(LowerMatrixIR.entriesFieldName -> MakeArray(FastSeq(), TArray(mr.typ.entryType)))))
    } else {
      val colsTable = TableRead(
        TableType(
          mr.typ.colType,
          FastIndexedSeq(),
          TStruct()
        ),
        dropRows = false,
        TableNativeReader(colsPath, spec.colsTableSpec(colsPath))
      )

      tr = TableMapGlobals(tr, InsertFields(
        Ref("global", tr.typ.globalType),
        FastSeq(LowerMatrixIR.colsFieldName -> GetField(TableCollect(colsTable), "rows"))
      ))

      val entries: TableIR = TableRead(
        TableType(
          TStruct(LowerMatrixIR.entriesFieldName -> TArray(mr.typ.entryType)),
          FastIndexedSeq(),
          TStruct()
        ),
        mr.dropRows,
        TableNativeReader(entriesPath, spec.entriesTableSpec(entriesPath))
      )

      tr = TableZipUnchecked(tr, entries)
    }

    tr
  }
}

case class MatrixRangeReader(nRows: Int, nCols: Int, nPartitions: Option[Int]) extends MatrixReader {
  val fullMatrixType: MatrixType = MatrixType.fromParts(
    globalType = TStruct.empty(),
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32()),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32()),
    entryType = TStruct.empty())

  override lazy val fullRVDType: RVDType = RVDType(
    PStruct("row_idx" -> PInt32(), MatrixType.entriesIdentifier -> PArray(PStruct())),
    FastIndexedSeq("row_idx"))

  val columnCount: Option[Int] = Some(nCols)

  lazy val partitionCounts: Option[IndexedSeq[Long]] = {
    val nPartitionsAdj = math.min(nRows, nPartitions.getOrElse(HailContext.get.sc.defaultParallelism))
    Some(partition(nRows, nPartitionsAdj).map(_.toLong))
  }

  override def lower(mr: MatrixRead): TableIR = {
    val uid1 = Symbol(genUID())

    TableRange(nRows, nPartitions.getOrElse(HailContext.get.sc.defaultParallelism))
      .rename(Map("idx" -> "row_idx"))
      .mapGlobals(makeStruct(LowerMatrixIR.colsField ->
        irRange(0, nCols).map('i ~> makeStruct('col_idx -> 'i))))
      .mapRows('row.insertFields(LowerMatrixIR.entriesField ->
        irRange(0, nCols).map('i ~> makeStruct())))
  }
}

case class MatrixRead(
  typ: MatrixType,
  dropCols: Boolean,
  dropRows: Boolean,
  reader: MatrixReader) extends MatrixIR {

  override lazy val rvdType: RVDType = reader.fullRVDType.subsetTo(typ.rvRowType)

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

  val typ: MatrixType = MatrixIR.filterCols(child.typ)._1

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

  val typ: MatrixType = MatrixIR.chooseColsWithArray(child.typ)._1

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

    child.typ.copyParts(colType = newColType, entryType = newEntryType)
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

  val typ: MatrixType = child.typ.copyParts(
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
    rvRowType = child.typ.rvRowType.updateKey(MatrixType.entriesIdentifier, TArray(entryExpr.typ)),
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
    child.typ.copy(rvRowType = child.typ.rvRowType.updateKey(MatrixType.entriesIdentifier, TArray(newEntries.typ)))

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

  val newRVRow = newRow.typ.asInstanceOf[TStruct].fieldOption(MatrixType.entriesIdentifier) match {
    case Some(f) =>
      assert(f.typ == child.typ.entryArrayType)
      newRow
    case None =>
      InsertFields(newRow, Seq(
        MatrixType.entriesIdentifier -> GetField(Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier)))
  }

  val typ: MatrixType = {
    child.typ.copy(rvRowType = newRVRow.typ.asInstanceOf[TStruct])
  }

  override lazy val rvdType: RVDType = RVDType(
      newRVRow.pType.asInstanceOf[PStruct],
      typ.rowKey)

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

  override lazy val rvdType: RVDType = child.rvdType

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = child.columnCount

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(prev.typ == child.typ)

    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val localNCols = prev.nCols
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val colValuesType = TArray(prev.typ.colType)
    val vaType = prev.rvd.rowPType
    val gType = MatrixType.getEntryType(vaType)

    var initOpNeedsSA = false
    var initOpNeedsGlobals = false
    var seqOpNeedsSA = false
    var seqOpNeedsGlobals = false

    val rewriteInitOp = { (nAggs: Int, initOp: IR) =>
      initOpNeedsSA = Mentions(initOp, "sa")
      initOpNeedsGlobals = Mentions(initOp, "global")
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case InitOp(i, args, aggSig) =>
            InitOp(
              ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(ir.Multiply(), ir.Ref(colIdx, TInt32()), ir.I32(nAggs)),
                i),
              args,
              aggSig)
          case _ =>
            ir.MapIR(rewrite)(x)
        }
      }

      val wrappedInit = if (initOpNeedsSA) {
        ir.Let(
          "sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref(colIdx, TInt32())),
          rewrite(initOp))
      } else {
        rewrite(initOp)
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
        colIdx,
        wrappedInit)
    }

    val rewriteSeqOp = { (nAggs: Int, seqOp: IR) =>
      seqOpNeedsSA = Mentions(seqOp, "sa")
      seqOpNeedsGlobals = Mentions(seqOp, "global")

      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case SeqOp(i, args, aggSig) =>
            SeqOp(
              ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(ir.Multiply(), ir.Ref(colIdx, TInt32()), ir.I32(nAggs)),
                i),
              args, aggSig)
          case _ =>
            ir.MapIR(rewrite)(x)
        }
      }

      var oneSampleSeqOp = ir.Let(
        "g",
        ir.ArrayRef(
          ir.GetField(ir.Ref("va", vaType.virtualType), MatrixType.entriesIdentifier),
          ir.Ref(colIdx, TInt32())),
        If(
          IsNA(ir.Ref("g", gType.virtualType)),
          Begin(FastSeq()),
          rewrite(seqOp))
      )

      if (seqOpNeedsSA)
        oneSampleSeqOp = ir.Let(
          "sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref(colIdx, TInt32())),
          oneSampleSeqOp)

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
        colIdx,
        oneSampleSeqOp)
    }

    val (entryAggs, initOps, seqOps, aggResultType, postAggIR) =
      ir.CompileWithAggregators[Long, Long, Long, Long, Long](
        "global", localGlobalsType.physicalType,
        "colValues", colValuesType.physicalType,
        "global", localGlobalsType.physicalType,
        "colValues", colValuesType.physicalType,
        "va", vaType,
        newCol, "AGGR",
        rewriteInitOp,
        rewriteSeqOp)

    var scanInitOpNeedsGlobals = false

    val (scanAggs, scanInitOps, scanSeqOps, scanResultType, postScanIR) =
      ir.CompileWithAggregators[Long, Long, Long, Long](
        "global", localGlobalsType.physicalType,
        "AGGR", aggResultType,
        "global", localGlobalsType.physicalType,
        "sa", prev.typ.colType.physicalType,
        CompileWithAggregators.liftScan(postAggIR), "SCANR",
        { (nAggs, init) =>
          scanInitOpNeedsGlobals = Mentions(init, "global")
          init
        },
        (nAggs, seq) => seq)

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long, Long](
      "AGGR", aggResultType,
      "SCANR", scanResultType,
      "global", localGlobalsType.physicalType,
      "sa", prev.typ.colType.physicalType,
      postScanIR)

    val nAggs = entryAggs.length

    assert(rTyp.virtualType == typ.colType, s"$rTyp, ${ typ.colType }")

    log.info(
      s"""MatrixMapCols: initOp ${ initOpNeedsGlobals } ${ initOpNeedsSA };
         |seqOp ${ seqOpNeedsGlobals } ${ seqOpNeedsSA }""".stripMargin)

    val depth = treeAggDepth(hc, prev.nPartitions)

    val colRVAggs = new Array[RegionValueAggregator](nAggs * localNCols)
    var i = 0
    while (i < localNCols) {
      var j = 0
      while (j < nAggs) {
        colRVAggs(i * nAggs + j) = entryAggs(j).newInstance()
        j += 1
      }
      i += 1
    }

    val aggResults = if (nAggs > 0) {
      Region.scoped { region =>
        val rvb: RegionValueBuilder = new RegionValueBuilder()
        rvb.set(region)

        val globals = if (initOpNeedsGlobals) {
          rvb.start(localGlobalsType.physicalType)
          rvb.addAnnotation(localGlobalsType, globalsBc.value)
          rvb.end()
        } else 0L

        val cols = if (initOpNeedsSA) {
          rvb.start(localColsType.physicalType)
          rvb.addAnnotation(localColsType, colValuesBc.value)
          rvb.end()
        } else 0L

        initOps(0)(region, colRVAggs, globals, false, cols, false)
      }

      type PC = (CompileWithAggregators.IRAggFun3[Long, Long, Long], Long, Long)
      prev.rvd.treeAggregateWithPartitionOp[PC, Array[RegionValueAggregator]](colRVAggs, { (i, ctx) =>
        val rvb = new RegionValueBuilder(ctx.freshRegion)

        val globals = if (seqOpNeedsGlobals) {
          rvb.start(localGlobalsType.physicalType)
          rvb.addAnnotation(localGlobalsType, globalsBc.value)
          rvb.end()
        } else 0L

        val cols = if (seqOpNeedsSA) {
          rvb.start(localColsType.physicalType)
          rvb.addAnnotation(localColsType, colValuesBc.value)
          rvb.end()
        } else 0L

        (seqOps(i), globals, cols)
      })({ case ((seqOpF, globals, cols), colRVAggs, rv) =>

        seqOpF(rv.region, colRVAggs, globals, false, cols, false, rv.offset, false)

        colRVAggs
      }, { (rvAggs1, rvAggs2) =>
        var i = 0
        while (i < rvAggs1.length) {
          rvAggs1(i).combOp(rvAggs2(i))
          i += 1
        }
        rvAggs1
      }, depth = depth)
    } else
      Array.empty[RegionValueAggregator]

    val prevColType = prev.typ.colType
    val rvb = new RegionValueBuilder()

    if (scanAggs.nonEmpty) {
      Region.scoped { region =>
        val rvb: RegionValueBuilder = new RegionValueBuilder()
        rvb.set(region)
        val globals = if (scanInitOpNeedsGlobals) {
          rvb.start(localGlobalsType.physicalType)
          rvb.addAnnotation(localGlobalsType, globalsBc.value)
          rvb.end()
        } else 0L

        scanInitOps(0)(region, scanAggs, globals, false)
      }
    }

    val colsF = f(0)
    val scanSeqOpF = scanSeqOps(0)

    val newColValues = Region.scoped { region =>
      rvb.set(region)
      rvb.start(localGlobalsType.physicalType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val globalRVoffset = rvb.end()

      val mapF = (a: Annotation, i: Int) => {

        rvb.start(aggResultType)
        rvb.startTuple()
        var j = 0
        while (j < nAggs) {
          aggResults(i * nAggs + j).result(rvb)
          j += 1
        }
        rvb.endTuple()
        val aggResultsOffset = rvb.end()

        val colRVb = new RegionValueBuilder(region)
        colRVb.start(prevColType.physicalType)
        colRVb.addAnnotation(prevColType, a)
        val colRVoffset = colRVb.end()

        rvb.start(scanResultType)
        rvb.startTuple()
        j = 0
        while (j < scanAggs.length) {
          scanAggs(j).result(rvb)
          j += 1
        }
        rvb.endTuple()
        val scanResultsOffset = rvb.end()

        val resultOffset = colsF(region, aggResultsOffset, false, scanResultsOffset, false, globalRVoffset, false, colRVoffset, false)
        scanSeqOpF(region, scanAggs, aggResultsOffset, false, globalRVoffset, false, colRVoffset, false)

        SafeRow(coerce[PStruct](rTyp), region, resultOffset)
      }
      BroadcastIndexedSeq(colValuesBc.value.zipWithIndex.map { case (a, i) => mapF(a, i) }, TArray(typ.colType), hc.sc)
    }

    prev.copy(typ = typ, colValues = newColValues)
  }
}

case class MatrixMapGlobals(child: MatrixIR, newGlobals: IR) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, newGlobals)

  val typ: MatrixType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  override lazy val rvdType: RVDType = child.rvdType

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

  override lazy val rvdType: RVDType = child.rvdType

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

  val typ: MatrixType = child.typ.copy(rvRowType = child.typ.rvRowType ++ TStruct(root -> table.typ.valueType))

  override lazy val rvdType: RVDType = child.rvdType.copy(
    rowType = child.rvdType.rowType.appendKey(
      root,
      table.rvdType.rowType.dropFields(table.typ.key.toSet)))

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

  private val rvRowType = child.typ.rvRowType

  val idx = Ref(genUID(), TInt32())
  val newRVRow: InsertFields = {
    val refs = path.init.scanLeft(Ref("va", rvRowType))((struct, name) =>
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

  val typ: MatrixType = child.typ.copy(rvRowType = newRVRow.typ)
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

object MatrixUnionRows {
  private def fixup(mir: MatrixIR): MatrixIR = {
    MatrixMapRows(mir,
      InsertFields(
        SelectFields(
          Ref("va", mir.typ.rvRowType),
          mir.typ.rvRowType.fieldNames.filter(_ != MatrixType.entriesIdentifier)
        ),
        Seq(MatrixType.entriesIdentifier -> GetField(Ref("va", mir.typ.rvRowType), MatrixType.entriesIdentifier))
      )
    )
  }

  def unify(mirs: IndexedSeq[MatrixIR]): IndexedSeq[MatrixIR] = mirs.map(fixup)
}

case class MatrixUnionRows(children: IndexedSeq[MatrixIR]) extends MatrixIR {
  require(children.length > 1)

  val typ = MatrixUnionRows.fixup(children.head).typ

  require(children.tail.forall(_.typ.rowKeyStruct == typ.rowKeyStruct))
  require(children.tail.forall(_.typ.rowType == typ.rowType))
  require(children.tail.forall(_.typ.entryType == typ.entryType))
  require(children.tail.forall(_.typ.colKeyStruct == typ.colKeyStruct))

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

  private val m = Map(entriesFieldName -> MatrixType.entriesIdentifier)

  child.typ.rowType.fieldType(entriesFieldName) match {
    case TArray(TStruct(_, _), _) =>
    case t => fatal(s"expected entry field to be an array of structs, found $t")
  }

  private val (colType, colsFieldIdx) = child.typ.globalType.field(colsFieldName) match {
    case Field(_, TArray(t@TStruct(_, _), _), idx) => (t, idx)
    case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
  }

  val typ: MatrixType = MatrixType(
    child.typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
    colKey,
    colType,
    child.typ.key,
    child.typ.rowType.rename(m))

  override lazy val rvdType: RVDType = child.rvdType.copy(rowType = child.rvdType.rowType.rename(m))

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

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val entries = GetField(Ref("row", child.typ.rowType), entriesFieldName)
    val cols = GetField(Ref("global", child.typ.globalType), colsFieldName)
    val checkedRow =
      ir.If(ir.IsNA(entries),
        ir.Die("missing entry array value in argument to CastTableToMatrix", child.typ.rowType),
        ir.If(ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.ArrayLen(entries), ir.ArrayLen(cols)),
          Ref("row", child.typ.rowType),
          Die("incorrect entry array length in argument to CastTableToMatrix", child.typ.rowType)))
    val checkedChild = ir.TableMapRows(child, checkedRow)

    checkedChild.execute(hc).toMatrixValue(colsFieldName, entriesFieldName, colKey)
  }
}

case class MatrixToMatrixApply(child: MatrixIR, function: MatrixToMatrixFunction) extends MatrixIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToMatrixApply(newChild, function)
  }

  override val (typ, rvdType) = function.typeInfo(child.typ, child.rvdType)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    function.execute(child.execute(hc))
  }
}

case class MatrixRename(child: MatrixIR,
  globalMap: Map[String, String], colMap: Map[String, String], rowMap: Map[String, String], entryMap: Map[String, String]) extends MatrixIR {
  require(globalMap.keys.forall(child.typ.globalType.hasField))
  require(colMap.keys.forall(child.typ.colType.hasField))
  require(rowMap.keys.forall(child.typ.rowType.hasField))
  require(entryMap.keys.forall(child.typ.entryType.hasField))

  lazy val typ: MatrixType = MatrixType.fromParts(
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

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    MatrixValue(typ, prev.globals, prev.colValues, prev.rvd.cast(rvdType.rowType))
  }
}