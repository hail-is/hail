package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.ir
import is.hail.expr.ir.functions.{MatrixToMatrixFunction, RelationalFunctions}
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PInt32, PStruct}
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
import org.json4s._

import scala.collection.mutable

object MatrixIR {
  def read(hc: HailContext, path: String, dropCols: Boolean = false, dropRows: Boolean = false, requestedType: Option[MatrixType]): MatrixIR = {
    val reader = MatrixNativeReader(path)
    MatrixRead(requestedType.getOrElse(reader.fullType), dropCols, dropRows, reader)
  }

  def range(hc: HailContext, nRows: Int, nCols: Int, nPartitions: Option[Int], dropCols: Boolean = false, dropRows: Boolean = false): MatrixIR = {
    val reader = MatrixRangeReader(nRows, nCols, nPartitions)
    MatrixRead(reader.fullType, dropCols = dropCols, dropRows = dropRows, reader = reader)
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
}

case class MatrixLiteral(value: MatrixValue) extends MatrixIR {
  val typ: MatrixType = value.typ

  override val rvdType: RVDType = value.rvd.typ

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

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
      classOf[MatrixBGENReader], classOf[MatrixPLINKReader], classOf[MatrixGENReader]))
}

abstract class MatrixReader {
  def apply(mr: MatrixRead): MatrixValue

  def columnCount: Option[Int]

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullType: MatrixType

  def fullRVDType: RVDType

  def requestType(requestedType: MatrixType): MatrixType = requestedType
}

case class MatrixNativeReader(path: String) extends MatrixReader {

  val spec: AbstractMatrixTableSpec = (RelationalSpec.read(HailContext.get, path): @unchecked) match {
    case mts: AbstractMatrixTableSpec => mts
    case _: AbstractTableSpec => fatal(s"file is a Table, not a MatrixTable: '$path'")
  }

  override val fullRVDType: RVDType = spec.rvdType(path)

  lazy val columnCount: Option[Int] = Some(RelationalSpec.read(HailContext.get, path + "/cols")
    .asInstanceOf[AbstractTableSpec]
    .partitionCounts
    .sum
    .toInt)

  lazy val partitionCounts: Option[IndexedSeq[Long]] = Some(spec.partitionCounts)

  val fullType = spec.matrix_type

  def apply(mr: MatrixRead): MatrixValue = {
    val hc = HailContext.get

    val requestedType = mr.typ
    assert(PruneDeadFields.isSupertype(requestedType, spec.matrix_type),
      s"R: ${ requestedType.parsableString() }\nS: ${ spec.matrix_type.parsableString() }")

    val globals = spec.globalsComponent.readLocal(hc, path, requestedType.globalType.physicalType)(0).asInstanceOf[Row]

    val colAnnotations =
      if (mr.dropCols)
        FastIndexedSeq.empty[Annotation]
      else
        spec.colsComponent.readLocal(hc, path, requestedType.colType.physicalType).asInstanceOf[IndexedSeq[Annotation]]

    val rvd =
      if (mr.dropRows)
        RVD.empty(hc.sc, requestedType.canonicalRVDType)
      else {
        val fullRowType = requestedType.rvRowType
        val localEntriesIndex = requestedType.entriesIdx

        val rowsRVD = spec.rowsComponent.read(hc, path, requestedType.rowType.physicalType)
        if (mr.dropCols) {
          val (t2, makeF) = ir.Compile[Long, Long](
            "row", requestedType.rowType.physicalType,
            MakeStruct(
              fullRowType.fields.zipWithIndex.map { case (f, i) =>
                val v: IR = if (i == localEntriesIndex)
                  MakeArray(FastSeq.empty, TArray(requestedType.entryType))
                else
                  GetField(Ref("row", requestedType.rowType), f.name)
                f.name -> v
              }))
          assert(t2.virtualType == fullRowType)

          rowsRVD.mapPartitionsWithIndex(requestedType.canonicalRVDType) { (i, it) =>
            val f = makeF(i)
            val rv2 = RegionValue()
            it.map { rv =>
              val off = f(rv.region, rv.offset, false)
              rv2.set(rv.region, off)
              rv2
            }
          }
        } else {
          val entriesRVD = spec.entriesComponent.read(hc, path, requestedType.entriesRVType.physicalType)
          val entriesRowType = entriesRVD.rowType

          val (t2, makeF) = ir.Compile[Long, Long, Long](
            "row", requestedType.rowType.physicalType,
            "entriesRow", entriesRowType.physicalType,
            MakeStruct(
              fullRowType.fields.zipWithIndex.map { case (f, i) =>
                val v: IR = if (i == localEntriesIndex)
                  GetField(Ref("entriesRow", entriesRowType), MatrixType.entriesIdentifier)
                else
                  GetField(Ref("row", requestedType.rowType), f.name)
                f.name -> v
              }))
          assert(t2.virtualType == fullRowType)

          rowsRVD.zipPartitionsWithIndex(requestedType.canonicalRVDType, entriesRVD) { (i, ctx, it1, it2) =>
            val f = makeF(i)
            val region = ctx.region
            val rv3 = RegionValue(region)
            new Iterator[RegionValue] {
              def hasNext = {
                val hn1 = it1.hasNext
                val hn2 = it2.hasNext
                assert(hn1 == hn2)
                hn1
              }

              def next(): RegionValue = {
                val rv1 = it1.next()
                val rv2 = it2.next()
                val off = f(region, rv1.offset, false, rv2.offset, false)
                rv3.set(region, off)
                rv3
              }
            }
          }
        }
      }

    MatrixValue(
      requestedType,
      BroadcastRow(globals, requestedType.globalType, hc.sc),
      BroadcastIndexedSeq(colAnnotations, TArray(requestedType.colType), hc.sc),
      rvd)
  }
}

case class MatrixRangeReader(nRows: Int, nCols: Int, nPartitions: Option[Int]) extends MatrixReader {
  val fullType: MatrixType = MatrixType.fromParts(
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

  def apply(mr: MatrixRead): MatrixValue = {
    assert(mr.typ == fullType)

    val partCounts = mr.partitionCounts.get.map(_.toInt)
    val nPartitionsAdj = mr.partitionCounts.get.length

    val hc = HailContext.get
    val localRVType = mr.rvdType.rowType
    val partStarts = partCounts.scanLeft(0)(_ + _)
    val localNCols = if (mr.dropCols) 0 else nCols

    val rvd = if (mr.dropRows)
      RVD.empty(hc.sc, fullType.canonicalRVDType)
    else {
      RVD(mr.rvdType,
        new RVDPartitioner(
          fullType.rowKeyStruct,
          Array.tabulate(nPartitionsAdj) { i =>
            val start = partStarts(i)
            Interval(Row(start), Row(start + partCounts(i)), includesStart = true, includesEnd = false)
          }),
        ContextRDD.parallelize[RVDContext](hc.sc, Range(0, nPartitionsAdj), nPartitionsAdj)
          .cmapPartitionsWithIndex { (i, ctx, _) =>
            val region = ctx.region
            val rvb = ctx.rvb
            val rv = RegionValue(region)

            val start = partStarts(i)
            Iterator.range(start, start + partCounts(i))
              .map { j =>
                rvb.start(localRVType)
                rvb.startStruct()

                // row idx field
                rvb.addInt(j)

                // entries field
                rvb.startArray(localNCols)
                var i = 0
                while (i < localNCols) {
                  rvb.startStruct()
                  rvb.endStruct()
                  i += 1
                }
                rvb.endArray()

                rvb.endStruct()
                rv.setOffset(rvb.end())
                rv
              }
          })
    }

    MatrixValue(fullType,
      BroadcastRow(Row(), fullType.globalType, hc.sc),
      BroadcastIndexedSeq(
        Iterator.range(0, localNCols)
          .map(Row(_))
          .toFastIndexedSeq,
        TArray(fullType.colType),
        hc.sc),
      rvd)
  }
}

case class MatrixRead(
  typ: MatrixType,
  dropCols: Boolean,
  dropRows: Boolean,
  reader: MatrixReader) extends MatrixIR {

  override lazy val rvdType: RVDType = reader.fullRVDType.subsetTo(typ.rvRowType)

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    MatrixRead(typ, dropCols, dropRows, reader)
  }

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val mv = reader(this)
    assert(mv.typ == typ)
    mv
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
}

case class MatrixFilterCols(child: MatrixIR, pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterCols = {
    assert(newChildren.length == 2)
    MatrixFilterCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = MatrixIR.filterCols(child.typ)._1

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts
}

case class MatrixFilterRows(child: MatrixIR, pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterRows = {
    assert(newChildren.length == 2)
    MatrixFilterRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  def typ: MatrixType = child.typ

  override def columnCount: Option[Int] = child.columnCount
}

case class MatrixChooseCols(child: MatrixIR, oldIndices: IndexedSeq[Int]) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixChooseCols = {
    assert(newChildren.length == 1)
    MatrixChooseCols(newChildren(0).asInstanceOf[MatrixIR], oldIndices)
  }

  val typ: MatrixType = MatrixIR.chooseColsWithArray(child.typ)._1

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override def columnCount: Option[Int] = Some(oldIndices.length)
}

case class MatrixCollectColsByKey(child: MatrixIR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

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

  def children: IndexedSeq[BaseIR] = Array(child, entryExpr, rowExpr)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateRowsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newRowExpr: IR) = newChildren
    MatrixAggregateRowsByKey(newChild, newEntryExpr, newRowExpr)
  }

  val typ: MatrixType = child.typ.copyParts(
    rowType = child.typ.rowKeyStruct ++ coerce[TStruct](rowExpr.typ),
    entryType = coerce[TStruct](entryExpr.typ)
  )

  override def columnCount: Option[Int] = child.columnCount

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val fullRowType = prev.rvd.rowPType

    val nCols = prev.nCols

    // Setup row aggregations
    val (rvAggsRow, makeInitRow, makeSeqRow, aggResultTypeRow, postAggIRRow) = ir.CompileWithAggregators[Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "global", child.typ.globalType.physicalType,
      "va", fullRowType,
      rowExpr, "AGGR",
      { (nAggs, initializeIR) => initializeIR },
      { (nAggs, sequenceIR) => sequenceIR }
    )

    val (rTypRow, makeAnnotateRow) = Compile[Long, Long, Long](
      "AGGR", aggResultTypeRow,
      "global", child.typ.globalType.physicalType,
      postAggIRRow)

    assert(coerce[TStruct](rTypRow.virtualType) == typ.rowValueStruct, s"$rTypRow, ${ typ.rowType }")

    // Setup entry aggregations
    val (minColType, minColValues, rewriteIR) = PruneDeadFields.pruneColValues(prev, entryExpr)

    val (rvAggsEntry, makeInitEntry, makeSeqEntry, aggResultTypeEntry, postAggIREntry) = ir.CompileWithAggregators[Long, Long, Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "colValues", TArray(minColType).physicalType,
      "global", child.typ.globalType.physicalType,
      "colValues", TArray(minColType).physicalType,
      "va", fullRowType,
      rewriteIR, "AGGR", { (nAggs, initializeIR) =>
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

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(nCols), ir.I32(1)),
          colIdx,
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", TArray(minColType)), ir.Ref(colIdx, TInt32())),
            rewrite(initializeIR)))
      }, { (nAggs, sequenceIR) =>
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

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(nCols), ir.I32(1)),
          colIdx,
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", TArray(minColType)), ir.Ref(colIdx, TInt32())),
            ir.Let("g", ir.ArrayRef(
              ir.GetField(ir.Ref("va", fullRowType.virtualType), MatrixType.entriesIdentifier),
              ir.Ref(colIdx, TInt32())),
              rewrite(sequenceIR))))
      })

    val (rTypEntry, makeAnnotateEntry) = Compile[Long, Long, Long, Long](
      "AGGR", aggResultTypeEntry,
      "global", child.typ.globalType.physicalType,
      "sa", minColType.physicalType,
      postAggIREntry)

    val nAggsEntry = rvAggsEntry.length

    assert(coerce[TStruct](rTypEntry.virtualType) == typ.entryType, s"$rTypEntry, ${ typ.entryType }")

    // Iterate through rows and aggregate
    val newRVType = typ.rvRowType
    val newRowKeyType = typ.rowKeyStruct
    val selectIdx = prev.typ.canonicalRVDType.kFieldIdx
    val keyOrd = prev.typ.canonicalRVDType.kRowOrd
    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(minColType).physicalType
    val colValuesBc = minColValues.broadcast
    val globalsBc = prev.globals.broadcast
    val newRVD = prev.rvd
      .repartition(prev.rvd.partitioner.strictify)
      .boundary
      .mapPartitionsWithIndex(typ.canonicalRVDType, { (i, ctx, it) =>
        val rvb = new RegionValueBuilder()
        val partRegion = ctx.freshRegion

        rvb.set(partRegion)
        rvb.start(localGlobalsType.physicalType)
        rvb.addAnnotation(localGlobalsType, globalsBc.value)
        val globals = rvb.end()

        rvb.start(localColsType)
        rvb.addAnnotation(localColsType.virtualType, colValuesBc.value)
        val cols = rvb.end()

        val initializeRow = makeInitRow(i)
        val sequenceRow = makeSeqRow(i)
        val annotateRow = makeAnnotateRow(i)

        val initializeEntry = makeInitEntry(i)
        val sequenceEntry = makeSeqEntry(i)
        val annotateEntry = makeAnnotateEntry(i)

        new Iterator[RegionValue] {
          var isEnd = false
          var current: RegionValue = _
          val rvRowKey: WritableRegionValue = WritableRegionValue(newRowKeyType.physicalType, ctx.freshRegion)
          val consumerRegion = ctx.region
          val newRV = RegionValue(consumerRegion)

          val colRVAggs = new Array[RegionValueAggregator](nAggsEntry * nCols)

          {
            var i = 0
            while (i < nCols) {
              var j = 0
              while (j < nAggsEntry) {
                colRVAggs(i * nAggsEntry + j) = rvAggsEntry(j).newInstance()
                j += 1
              }
              i += 1
            }
          }

          def hasNext: Boolean = {
            if (isEnd || (current == null && !it.hasNext)) {
              isEnd = true
              return false
            }
            if (current == null)
              current = it.next()
            true
          }

          def next(): RegionValue = {
            if (!hasNext)
              throw new java.util.NoSuchElementException()

            rvRowKey.setSelect(fullRowType, selectIdx, current)

            colRVAggs.foreach(_.clear())
            rvAggsRow.foreach(_.clear())

            initializeEntry(current.region, colRVAggs, globals, false, cols, false)
            initializeRow(current.region, rvAggsRow, globals, false)

            do {
              sequenceEntry(current.region, colRVAggs,
                globals, false,
                cols, false,
                current.offset, false)

              sequenceRow(current.region, rvAggsRow,
                globals, false,
                current.offset, false)

              current = null
            } while (hasNext && keyOrd.equiv(rvRowKey.value, current))

            rvb.set(consumerRegion)

            val rowAggResultsOffset = {
              rvb.start(aggResultTypeRow)
              rvb.startTuple()
              var j = 0
              while (j < rvAggsRow.length) {
                rvAggsRow(j).result(rvb)
                j += 1
              }
              rvb.endTuple()
              rvb.end()
            }

            val entryAggResultsOffsets = Array.tabulate(nCols) { i =>
              rvb.start(aggResultTypeEntry)
              rvb.startTuple()
              var j = 0
              while (j < nAggsEntry) {
                colRVAggs(i * nAggsEntry + j).result(rvb)
                j += 1
              }
              rvb.endTuple()
              rvb.end()
            }

            rvb.start(newRVType.physicalType)
            rvb.startStruct()

            {
              rvb.addAllFields(newRowKeyType.physicalType, rvRowKey.value)

              val newRowOff = annotateRow(consumerRegion,
                rowAggResultsOffset, false,
                globals, false)
              rvb.addAllFields(coerce[PStruct](rTypRow), consumerRegion, newRowOff)
            }

            rvb.startArray(nCols)

            {
              var i = 0
              while (i < nCols) {
                val newEntryOff = annotateEntry(consumerRegion,
                  entryAggResultsOffsets(i), false,
                  globals, false,
                  localColsType.loadElement(consumerRegion, cols, i), localColsType.isElementMissing(consumerRegion, cols, i))

                rvb.addRegionValue(rTypEntry, consumerRegion, newEntryOff)

                i += 1
              }
            }
            rvb.endArray()
            rvb.endStruct()
            newRV.setOffset(rvb.end())
            newRV
          }
        }
      })

    prev.copy(rvd = newRVD, typ = typ)
  }
}

case class MatrixAggregateColsByKey(child: MatrixIR, entryExpr: IR, colExpr: IR) extends MatrixIR {
  require(child.typ.colKey.nonEmpty)

  def children: IndexedSeq[BaseIR] = Array(child, entryExpr, colExpr)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateColsByKey = {
    val IndexedSeq(newChild: MatrixIR, newEntryExpr: IR, newColExpr: IR) = newChildren
    MatrixAggregateColsByKey(newChild, newEntryExpr, newColExpr)
  }

  val typ = child.typ.copyParts(
    entryType = coerce[TStruct](entryExpr.typ),
    colType = child.typ.colKeyStruct ++ coerce[TStruct](colExpr.typ))

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val mv = child.execute(hc)

    // local things for serialization
    val oldNCols = mv.nCols
    val oldRVRowType = mv.rvd.rowPType
    val oldEntriesIndex = MatrixType.getEntriesIndex(oldRVRowType)
    val oldColsType = TArray(mv.typ.colType)
    val oldColValues = mv.colValues
    val oldColValuesBc = mv.colValues.broadcast
    val oldGlobalsBc = mv.globals.broadcast
    val oldGlobalsType = mv.typ.globalType

    val newRVType = typ.rvRowType.physicalType
    val newColType = typ.colType
    val newColKeyType = typ.colKeyStruct
    val newColValueType = typ.colValueStruct

    val keyIndices = mv.typ.colKey.map(k => mv.typ.colType.field(k).index)
    val keys = oldColValuesBc.value.map { a => Row.fromSeq(keyIndices.map(a.asInstanceOf[Row].get)) }.toSet.toArray
    val nKeys = keys.length

    val keysByColumn = oldColValues.value.map { sa => Row.fromSeq(keyIndices.map(sa.asInstanceOf[Row].get)) }
    val keyMap = keys.zipWithIndex.toMap
    val newColumnIndices = keysByColumn.map { k => keyMap(k) }.toArray
    val newColumnIndicesType = TArray(TInt32())

    // Column aggregations

    val (rvAggsColTemplate, makeInitCol, makeSeqCol, aggResultTypeCol, postAggIRCol) = ir.CompileWithAggregators[Long, Long, Long, Long](
      "global", oldGlobalsType.physicalType,
      "global", oldGlobalsType.physicalType,
      "colValues", oldColsType.physicalType,
      "newColumnIndices", newColumnIndicesType.physicalType,
      colExpr, "AGGR",
      { (nAggs, initializeIR) =>
        val colIdx = ir.genUID()

        def rewrite(x: IR): IR = {
          x match {
            case InitOp(i, args, aggSig) =>
              InitOp(ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(
                  ir.Multiply(),
                  ir.Ref(colIdx, TInt32()),
                  ir.I32(nAggs)),
                i),
                args,
                aggSig)
            case _ =>
              ir.MapIR(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(nKeys), ir.I32(1)),
          colIdx,
          rewrite(initializeIR))
      },
      { (nAggs, sequenceIR) =>
        val colIdx = ir.genUID()

        def rewrite(x: IR): IR = {
          x match {
            case SeqOp(i, args, aggSig) =>
              SeqOp(
                ir.ApplyBinaryPrimOp(ir.Add(),
                  ir.ApplyBinaryPrimOp(
                    ir.Multiply(),
                    ir.ArrayRef(ir.Ref("newColumnIndices", newColumnIndicesType), ir.Ref(colIdx, TInt32())),
                    ir.I32(nAggs)),
                  i),
                args, aggSig)
            case _ =>
              ir.MapIR(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(oldNCols), ir.I32(1)),
          colIdx,
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", oldColsType), ir.Ref(colIdx, TInt32())),
            rewrite(sequenceIR)
          ))
      }
    )

    val (rTypCol, makeAnnotateCol) = ir.Compile[Long, Long, Long](
      "AGGR", aggResultTypeCol,
      "global", oldGlobalsType.physicalType,
      postAggIRCol
    )
    assert(coerce[TStruct](rTypCol.virtualType) == newColValueType)

    val nAggsCol = rvAggsColTemplate.length
    val rvAggsCol = new Array[RegionValueAggregator](nAggsCol * nKeys)
    var i = 0
    while (i < nKeys) {
      var j = 0
      while (j < nAggsCol) {
        rvAggsCol(i * nAggsCol + j) = rvAggsColTemplate(j).newInstance()
        j += 1
      }
      i += 1
    }

    val rvb = new RegionValueBuilder()

    val newColValues = Region.scoped { region =>
      rvb.set(region)

      val globals = {
        rvb.start(oldGlobalsType.physicalType)
        rvb.addAnnotation(oldGlobalsType, oldGlobalsBc.value)
        rvb.end()
      }

      val cols = {
        rvb.start(oldColsType.physicalType)
        rvb.addAnnotation(oldColsType, oldColValuesBc.value)
        rvb.end()
      }

      val colIndices = {
        rvb.start(newColumnIndicesType.physicalType)
        rvb.startArray(newColumnIndices.length)
        var i = 0
        while (i < newColumnIndices.length) {
          rvb.addInt(newColumnIndices(i))
          i += 1
        }
        rvb.endArray()
        rvb.end()
      }

      makeInitCol(0)(region, rvAggsCol, globals, false)
      makeSeqCol(0)(region, rvAggsCol, globals, false, cols, false, colIndices, false)

      val annotateF = makeAnnotateCol(0)

      BroadcastIndexedSeq(keys.zipWithIndex.map { case (a: Annotation, i: Int) =>
        val aggResults = {
          rvb.start(aggResultTypeCol)
          rvb.startTuple()
          var j = 0
          while (j < nAggsCol) {
            rvAggsCol(i * nAggsCol + j).result(rvb)
            j += 1
          }
          rvb.endTuple()
          rvb.end()
        }

        val colKeys = {
          rvb.start(newColKeyType.physicalType)
          rvb.addAnnotation(newColKeyType, a)
          rvb.end()
        }

        val colValues = annotateF(region, aggResults, false, globals, false)

        val result = {
          rvb.start(newColType.physicalType)
          rvb.startStruct()
          rvb.addAllFields(newColKeyType.physicalType, region, colKeys)
          rvb.addAllFields(newColValueType.physicalType, region, colValues)
          rvb.endStruct()
          rvb.end()
        }

        SafeRow(typ.colType.physicalType, region, result)
      }, TArray(typ.colType), hc.sc)
    }

    // Entry aggregations

    val transformInitOp: (Int, IR) => IR = { (nAggs, initOpIR) =>
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case InitOp(i, args, aggSig) =>
            InitOp(ir.ApplyBinaryPrimOp(ir.Add(),
              ir.ApplyBinaryPrimOp(
                ir.Multiply(),
                ir.Ref(colIdx, TInt32()),
                ir.I32(nAggs)),
              i),
              args,
              aggSig)
          case _ =>
            ir.MapIR(rewrite)(x)
        }
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(nKeys), ir.I32(1)),
        colIdx,
        rewrite(initOpIR))
    }

    val transformSeqOp: (Int, IR) => IR = { (nAggs, seqOpIR) =>
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case SeqOp(i, args, aggSig) =>
            SeqOp(
              ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(
                  ir.Multiply(),
                  ir.ArrayRef(ir.Ref("newColumnIndices", newColumnIndicesType), ir.Ref(colIdx, TInt32())),
                  ir.I32(nAggs)),
                i),
              args, aggSig)
          case _ =>
            ir.MapIR(rewrite)(x)
        }
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(oldNCols), ir.I32(1)),
        colIdx,
        ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", oldColsType), ir.Ref(colIdx, TInt32())),
          ir.Let("g", ir.ArrayRef(
            ir.GetField(ir.Ref("va", oldRVRowType.virtualType), MatrixType.entriesIdentifier),
            ir.Ref(colIdx, TInt32())),
            rewrite(seqOpIR)
          )))
    }

    val (rvAggsEntryTemplate, makeInitEntry, makeSeqEntry, aggResultTypeEntry, postAggIREntry) = ir.CompileWithAggregators[Long, Long, Long, Long, Long, Long](
      "global", oldGlobalsType.physicalType,
      "va", oldRVRowType,
      "global", oldGlobalsType.physicalType,
      "colValues", oldColsType.physicalType,
      "va", oldRVRowType,
      "newColumnIndices", newColumnIndicesType.physicalType,
      entryExpr, "AGGR",
      transformInitOp,
      transformSeqOp)

    val (rTypEntry, makeAnnotateEntry) = ir.Compile[Long, Long, Long, Long](
      "AGGR", aggResultTypeEntry,
      "global", oldGlobalsType.physicalType,
      "va", oldRVRowType,
      postAggIREntry
    )
    assert(rTypEntry.virtualType == typ.entryType)

    val nAggsEntry = rvAggsEntryTemplate.length
    val rvAggsEntry = new Array[RegionValueAggregator](nAggsEntry * nKeys)
    i = 0
    while (i < nKeys) {
      var j = 0
      while (j < nAggsEntry) {
        rvAggsEntry(i * nAggsEntry + j) = rvAggsEntryTemplate(j).newInstance()
        j += 1
      }
      i += 1
    }

    val mapPartitionF = { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
      val rvb = new RegionValueBuilder()

      val partitionRegion = ctx.freshRegion

      rvb.set(partitionRegion)
      rvb.start(oldGlobalsType.physicalType)
      rvb.addAnnotation(oldGlobalsType, oldGlobalsBc.value)
      val partitionWideGlobalsOffset = rvb.end()

      rvb.start(oldColsType.physicalType)
      rvb.addAnnotation(oldColsType, oldColValuesBc.value)
      val partitionWideColumnsOffset = rvb.end()

      rvb.start(newColumnIndicesType.physicalType)
      rvb.startArray(newColumnIndices.length)
      var i = 0
      while (i < newColumnIndices.length) {
        rvb.addInt(newColumnIndices(i))
        i += 1
      }
      rvb.endArray()
      val partitionWideMapOffset = rvb.end()

      val initEntry = makeInitEntry(i)
      val sequenceEntry = makeSeqEntry(i)
      val annotateEntry = makeAnnotateEntry(i)

      it.map { rv =>
        val oldRow = rv.offset

        rvb.set(rv.region)
        rvb.start(oldGlobalsType.physicalType)
        rvb.addRegionValue(oldGlobalsType.physicalType, partitionRegion, partitionWideGlobalsOffset)
        val globalsOffset = rvb.end()

        rvb.set(rv.region)
        rvb.start(oldColsType.physicalType)
        rvb.addRegionValue(oldColsType.physicalType, partitionRegion, partitionWideColumnsOffset)
        val columnsOffset = rvb.end()

        rvb.set(rv.region)
        rvb.start(newColumnIndicesType.physicalType)
        rvb.addRegionValue(newColumnIndicesType.physicalType, partitionRegion, partitionWideMapOffset)
        val mapOffset = rvb.end()

        var j = 0
        while (j < rvAggsEntry.length) {
          rvAggsEntry(j).clear()
          j += 1
        }

        initEntry(rv.region, rvAggsEntry, globalsOffset, false, oldRow, false)
        sequenceEntry(rv.region, rvAggsEntry, globalsOffset, false, columnsOffset, false, oldRow, false, mapOffset, false)

        val resultOffsets = Array.tabulate(nKeys) { i =>
          var j = 0
          rvb.start(aggResultTypeEntry)
          rvb.startTuple()
          while (j < nAggsEntry) {
            rvAggsEntry(i * nAggsEntry + j).result(rvb)
            j += 1
          }
          rvb.endTuple()
          val aggResultOffset = rvb.end()
          annotateEntry(rv.region, aggResultOffset, false, globalsOffset, false, oldRow, false)
        }

        rvb.start(newRVType)
        rvb.startStruct()
        var k = 0
        while (k < newRVType.size) {
          if (k != oldEntriesIndex)
            rvb.addField(oldRVRowType, rv, k)
          k += 1
        }

        i = 0
        rvb.startArray(nKeys)
        while (i < nKeys) {
          rvb.addRegionValue(rTypEntry, rv.region, resultOffsets(i))
          i += 1
        }
        rvb.endArray()
        rvb.endStruct()
        rv.setOffset(rvb.end())
        rv
      }
    }

    val newRVD = mv.rvd.mapPartitionsWithIndex(mv.rvd.typ.copy(rowType = newRVType), mapPartitionF)
    mv.copy(typ = typ, colValues = newColValues, rvd = newRVD)
  }
}

case class MatrixUnionCols(left: MatrixIR, right: MatrixIR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(left, right)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixUnionCols = {
    assert(newChildren.length == 2)
    MatrixUnionCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[MatrixIR])
  }

  val typ: MatrixType = left.typ

  override def columnCount: Option[Int] =
    left.columnCount.flatMap(leftCount => right.columnCount.map(rightCount => leftCount + rightCount))
}

case class MatrixMapEntries(child: MatrixIR, newEntries: IR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child, newEntries)

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

  def children: IndexedSeq[BaseIR] = Array(child, newRow)

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
  def children: IndexedSeq[BaseIR] = Array(child, newCol)

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

      var oneSampleSeqOp = ir.Let("g", ir.ArrayRef(
        ir.GetField(ir.Ref("va", vaType.virtualType), MatrixType.entriesIdentifier),
        ir.Ref(colIdx, TInt32())),
        rewrite(seqOp)
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

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child, table)

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

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    val tab = table.execute(hc)

    val keyTypes = tab.typ.keyType.types
    val colKeyTypes = prev.typ.colKeyStruct.types

    val keyedRDD = tab.keyedRDD().filter { case (k, _) => !k.anyNull }

    assert(keyTypes.length == colKeyTypes.length
      && keyTypes.zip(colKeyTypes).forall { case (l, r) => l.isOfType(r) },
      s"MT col key: ${ colKeyTypes.mkString(", ") }, TB key: ${ keyTypes.mkString(", ") }")
    val r = keyedRDD.map { case (k, v) => (k: Annotation, v: Annotation) }

    val m = r.collectAsMap()
    val colKeyF = prev.typ.extractColKey

    val newAnnotations = prev.colValues.value
      .map { row =>
        val key = colKeyF(row.asInstanceOf[Row])
        val newAnnotation = inserter(row, m.getOrElse(key, null))
        newAnnotation
      }
    prev.copy(typ = typ, colValues = BroadcastIndexedSeq(newAnnotations, TArray(colType), hc.sc))
  }
}

case class MatrixAnnotateRowsTable(
  child: MatrixIR,
  table: TableIR,
  root: String,
  key: Option[IndexedSeq[IR]]) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child, table) ++ key.getOrElse(FastIndexedSeq.empty[IR])

  override def columnCount: Option[Call] = child.columnCount

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  val typ: MatrixType = child.typ.copy(rvRowType = child.typ.rvRowType ++ TStruct(root -> table.typ.valueType))

  override lazy val rvdType: RVDType = child.rvdType.copy(
    rowType = child.rvdType.rowType.appendKey(
      root,
      table.rvdType.rowType.dropFields(table.typ.key.toSet)))

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAnnotateRowsTable = {
    val (child: MatrixIR) +: (table: TableIR) +: newKey = newChildren
    MatrixAnnotateRowsTable(
      child, table,
      root,
      key.map { keyIRs =>
        assert(newKey.length == keyIRs.length)
        newKey.map(_.asInstanceOf[IR])
      }
    )
  }

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    val tv = table.execute(hc)
    key match {
      // annotateRowsIntervals
      case None if table.typ.keyType.size == 1
        && table.typ.keyType.types(0) == TInterval(child.typ.rowKeyStruct.types(0)) =>
        val (newRVPType, ins) =
          prev.rvd.rowPType.unsafeStructInsert(table.typ.valueType.physicalType, List(root))

        val rightRVDType = tv.rvd.typ
        val leftRVDType = child.typ.canonicalRVDType

        val zipper = { (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
          val rvb = new RegionValueBuilder()
          val rv2 = RegionValue()
          OrderedRVIterator(leftRVDType, it, ctx).leftIntervalJoinDistinct(
            OrderedRVIterator(rightRVDType, intervals, ctx)
          )
            .map { case Muple(rv, i) =>
              rvb.set(rv.region)
              rvb.start(newRVPType)
              ins(
                rv.region,
                rv.offset,
                rvb,
                () => if (i == null) rvb.setMissing() else rvb.selectRegionValue(rightRVDType.rowType, rightRVDType.valueFieldIdx, i))
              rv2.set(rv.region, rvb.end())

              rv2
            }
        }

        val newMatrixType = child.typ.copy(rvRowType = newRVPType.virtualType)
        val newRVD = prev.rvd.intervalAlignAndZipPartitions(RVDType(newRVPType, newMatrixType.rowKey), tv.rvd)(zipper)
        prev.copy(typ = typ, rvd = newRVD)

      // annotateRowsTable using non-key MT fields
      case Some(newKeys) =>
        // FIXME: here be monsters

        // used to zipWithIndex in multiple places
        val partitionCounts = child.getOrComputePartitionCounts()

        val prevRowKeys = child.typ.rowKey.toArray
        val newKeyUIDs = Array.fill(newKeys.length)(ir.genUID())
        val indexUID = ir.genUID()

        // has matrix row key and foreign join key
        val mrt = Interpret(
          MatrixRowsTable(
            MatrixMapRows(
              child,
              MakeStruct(
                prevRowKeys.zip(
                  prevRowKeys.map(rk => GetField(Ref("va", child.typ.rvRowType), rk))
                ) ++ newKeyUIDs.zip(newKeys)))))
        val indexedRVD1 = mrt.rvd
          .zipWithIndex(indexUID, Some(partitionCounts))
        val tl1 = TableLiteral(mrt.copy(typ = mrt.typ.copy(rowType = indexedRVD1.rowType), rvd = indexedRVD1))

        // ordered by foreign key, filtered to remove null keys
        val sortedTL = Interpret(
          TableKeyBy(
            TableFilter(tl1,
              ApplyUnaryPrimOp(
                Bang(),
                newKeyUIDs
                  .map(k => IsNA(GetField(Ref("row", mrt.typ.rowType), k)))
                  .reduce[IR] { case (l, r) => ApplySpecial("||", FastIndexedSeq(l, r)) })),
            newKeyUIDs))

        val left = sortedTL.rvd
        val right = tv.rvd
        val joined = left.orderedLeftJoinDistinctAndInsert(right, root)

        // At this point 'joined' is sorted by the foreign key, so need to resort by row key
        // first, change the partitioner to include the index field in the key so the shuffled result is sorted by index
        val extendedKey = prev.rvd.typ.key ++ Array(indexUID)
        val rpJoined = joined.repartition(
          prev.rvd.partitioner.extendKey(joined.typ.copy(key = extendedKey).kType.virtualType),
          shuffle = true)

        // the lift and dropLeft flags are used to optimize some of the struct manipulation operations
        val newRVD = prev.rvd.zipWithIndex(indexUID, Some(partitionCounts))
          .extendKeyPreservesPartitioning(prev.rvd.typ.key ++ Array(indexUID))
          .orderedLeftJoinDistinctAndInsert(rpJoined, root, lift = Some(root), dropLeft = Some(Array(indexUID)))
        MatrixValue(typ, prev.globals, prev.colValues, newRVD)

      // annotateRowsTable using key
      case None =>
        assert(child.typ.rowKeyStruct.types.zip(table.typ.keyType.types).forall { case (l, r) => l.isOfType(r) })
        val newRVD = prev.rvd.orderedLeftJoinDistinctAndInsert(
          tv.rvd, root)
        prev.copy(typ = typ, rvd = newRVD)
    }
  }
}

case class TableToMatrixTable(
  child: TableIR,
  rowKey: IndexedSeq[String],
  colKey: IndexedSeq[String],
  rowFields: IndexedSeq[String],
  colFields: IndexedSeq[String],
  nPartitions: Option[Int] = None
) extends MatrixIR {
  // no fields used twice
  private val fieldsUsed = mutable.Set.empty[String]
  (rowKey ++ colKey ++ rowFields ++ colFields).foreach { f =>
    assert(!fieldsUsed.contains(f))
    fieldsUsed += f
  }

  // need keys for rows and cols
  assert(rowKey.nonEmpty)
  assert(colKey.nonEmpty)

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableToMatrixTable = {
    val IndexedSeq(newChild) = newChildren
    TableToMatrixTable(
      newChild.asInstanceOf[TableIR],
      rowKey,
      colKey,
      rowFields,
      colFields,
      nPartitions
    )
  }

  private val rowType = TStruct((rowKey ++ rowFields).map(f => f -> child.typ.rowType.field(f).typ): _*)
  private val colType = TStruct((colKey ++ colFields).map(f => f -> child.typ.rowType.field(f).typ): _*)
  private val entryFields = child.typ.rowType.fieldNames.filter(f => !fieldsUsed.contains(f))
  private val entryType = TStruct(entryFields.map(f => f -> child.typ.rowType.field(f).typ): _*)

  val typ: MatrixType = MatrixType.fromParts(
    child.typ.globalType,
    colKey,
    colType,
    rowKey,
    rowType,
    entryType)

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val colKeyIndices = colKey.map(child.typ.rowType.fieldIdx(_)).toArray
    val colValueIndices = colFields.map(child.typ.rowType.fieldIdx(_)).toArray
    val localRowPType = prev.rvd.rowPType
    val localRowType = localRowPType.virtualType
    val localColData = prev.rvd.mapPartitions { it =>
      it.map { rv =>
        val colKey = SafeRow.selectFields(localRowPType, rv)(colKeyIndices)
        val colValues = SafeRow.selectFields(localRowPType, rv)(colValueIndices)
        colKey -> colValues
      }
    }.reduceByKey({ case (l, _) => l }) // poor man's distinctByKey
      .collect()

    val nCols = localColData.length

    val colIndexBc = hc.sc.broadcast(localColData.zipWithIndex
      .map { case ((k, _), i) => (k, i) }
      .toMap)

    val colDataConcat = localColData.map { case (keys, values) => Row.fromSeq(keys.toSeq ++ values.toSeq): Annotation }
    val colKeysBc = hc.sc.broadcast(localColData.map(_._1))

    // allFieldIndices has all row + entry fields
    val allFieldIndices = rowKey.map(localRowType.fieldIdx(_)) ++
      rowFields.map(localRowType.fieldIdx(_)) ++
      entryFields.map(localRowType.fieldIdx(_))

    // FIXME replace with field namespaces
    val INDEX_UID = "*** COL IDX ***"

    // row and entry fields, plus an integer index
    val rowEntryStruct = rowType ++ entryType ++ TStruct(INDEX_UID -> TInt32Optional)
    val rowEntryStructPtype = rowEntryStruct.physicalType
    val rowKeyIndices = rowKey.map(rowEntryStruct.fieldIdx)
    val rowKeyF: Row => Row = r => Row.fromSeq(rowKeyIndices.map(r.get))

    val rowEntryRVD = prev.rvd.mapPartitions(RVDType(rowEntryStructPtype, FastIndexedSeq())) { it =>
      val ur = new UnsafeRow(localRowPType)
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { rv =>
        rvb.set(rv.region)

        rvb.start(rowEntryStructPtype)
        rvb.startStruct()

        // add all non-col fields
        var i = 0
        while (i < allFieldIndices.length) {
          rvb.addField(localRowPType, rv, allFieldIndices(i))
          i += 1
        }

        // look up col key, replace with int index
        ur.set(rv)
        val colKey = Row.fromSeq(colKeyIndices.map(ur.get))
        val idx = colIndexBc.value(colKey)
        rvb.addInt(idx)

        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    val ordType = RVDType(rowEntryStructPtype, rowKey ++ FastIndexedSeq(INDEX_UID))
    val ordTypeNoIndex = RVDType(rowEntryStructPtype, rowKey)
    val ordered = rowEntryRVD.changeKey(ordType.key, rowKey.length)
    val orderedEntryIndices = entryFields.map(rowEntryStruct.fieldIdx)
    val orderedRowIndices = (rowKey ++ rowFields).map(rowEntryStruct.fieldIdx)

    val idxIndex = rowEntryStruct.fieldIdx(INDEX_UID)
    assert(idxIndex == rowEntryStruct.size - 1)

    val newRVType: PStruct = typ.rvRowType.physicalType

    val newRVD = ordered.boundary.mapPartitions(RVDType(newRVType, typ.rowKey), { (ctx, it) =>
      val region = ctx.region
      val rvb = ctx.rvb
      val outRV = RegionValue(region)

      OrderedRVIterator(
        ordTypeNoIndex,
        it,
        ctx
      ).staircase.map { rowIt =>
        rvb.start(newRVType)
        rvb.startStruct()
        var i = 0
        while (i < orderedRowIndices.length) {
          rvb.addField(rowEntryStructPtype, rowIt.value, orderedRowIndices(i))
          i += 1
        }
        rvb.startArray(nCols)
        i = 0
        var lastSeen = -1
        for (rv <- rowIt) {
          val nextInt = rv.region.loadInt(rowEntryStructPtype.loadField(rv.region, rv.offset, idxIndex))
          if (nextInt == lastSeen) // duplicate (RK, CK) pair
            fatal(s"'to_matrix_table': duplicate (row key, col key) pairs are not supported\n" +
              s"  Row key: ${ rowKeyF(new UnsafeRow(rowEntryStructPtype, rv)) }\n" +
              s"  Col key: ${ colKeysBc.value(nextInt) }")
          lastSeen = nextInt
          while (i < nextInt) {
            rvb.setMissing()
            i += 1
          }
          rvb.startStruct()
          var j = 0
          while (j < orderedEntryIndices.length) {
            rvb.addField(rowEntryStruct.physicalType, rv, orderedEntryIndices(j))
            j += 1
          }
          rvb.endStruct()
          i += 1
        }
        while (i < nCols) {
          rvb.setMissing()
          i += 1
        }
        rvb.endArray()
        rvb.endStruct()
        outRV.setOffset(rvb.end())
        outRV
      }
    })
    MatrixValue(
      typ,
      prev.globals,
      BroadcastIndexedSeq(colDataConcat, TArray(colType), hc.sc),
      newRVD)
  }
}

case class MatrixExplodeRows(child: MatrixIR, path: IndexedSeq[String]) extends MatrixIR {
  assert(path.nonEmpty)

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

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

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

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

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixDistinctByRow = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixDistinctByRow(newChild)
  }

  override def columnCount: Option[Int] = child.columnCount
}

case class MatrixExplodeCols(child: MatrixIR, path: IndexedSeq[String]) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

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

  protected[ir] override def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val fullRowType = prev.rvd.rowPType

    var size = 0
    val keys = prev.colValues.value.map { sa =>
      val ks = querier(sa).asInstanceOf[Iterable[Any]]
      if (ks == null)
        Iterable.empty[Any]
      else {
        size += ks.size
        ks
      }
    }

    val sampleMap = new Array[Int](size)
    val newColValues = new Array[Annotation](size)
    val newNCols = newColValues.length

    var i = 0
    var j = 0
    while (i < prev.nCols) {
      keys(i).foreach { e =>
        sampleMap(j) = i
        newColValues(j) = inserter(prev.colValues.value(i), e)
        j += 1
      }
      i += 1
    }

    val sampleMapBc = hc.sc.broadcast(sampleMap)
    val localEntriesIndex = MatrixType.getEntriesIndex(fullRowType)
    val localEntriesType = MatrixType.getEntryArrayType(fullRowType)
    val localEntryType = MatrixType.getEntryType(fullRowType)

    prev.insertEntries(noOp, newColType = newColType,
      newColValues = prev.colValues.copy(value = newColValues, t = TArray(newColType)))(
      localEntryType,
      { case (_, rv, rvb) =>

        val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)
        rvb.startArray(newNCols)
        var i = 0
        while (i < newNCols) {
          rvb.addElement(localEntriesType, rv.region, entriesOffset, sampleMapBc.value(i))
          i += 1
        }
        rvb.endArray()
      })
  }
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

  private val newRowType = child.typ.rowType.rename(m)

  val typ: MatrixType = MatrixType(
    child.typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
    colKey,
    colType,
    child.typ.key,
    newRowType)

  override lazy val rvdType: RVDType = child.rvdType.copy(rowType = child.rvdType.rowType.rename(m))

  def children: IndexedSeq[BaseIR] = Array(child)

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

    val prev = checkedChild.execute(hc)

    val colValues = prev.globals.value.getAs[IndexedSeq[Annotation]](colsFieldIdx)
    val newGlobals = {
      val (pre, post) = prev.globals.value.toSeq.splitAt(colsFieldIdx)
      Row.fromSeq(pre ++ post.tail)
    }

    val newRVD = prev.rvd.cast(prev.rvd.rowPType.rename(m))

    MatrixValue(
      typ,
      BroadcastRow(newGlobals, typ.globalType, hc.sc),
      BroadcastIndexedSeq(colValues, TArray(typ.colType), hc.sc),
      newRVD
    )
  }
}

case class MatrixToMatrixApply(child: MatrixIR, function: MatrixToMatrixFunction) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

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
