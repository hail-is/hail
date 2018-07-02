package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.types._
import is.hail.expr.{TableAnnotationImpex, ir}
import is.hail.io._
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods
import org.json4s.jackson.JsonMethods.parse

object MatrixIR {
  def chooseColsWithArray(typ: MatrixType): (MatrixType, (MatrixValue, Array[Int]) => MatrixValue) = {
    val rowType = typ.rvRowType
    val keepType = TArray(+TInt32())
    val (rTyp, makeF) = ir.Compile[Long, Long, Long]("row", rowType,
      "keep", keepType,
      body = InsertFields(ir.Ref("row", rowType), Seq((MatrixType.entriesIdentifier,
        ir.ArrayMap(ir.Ref("keep", keepType), "i",
          ir.ArrayRef(ir.GetField(ir.In(0, rowType), MatrixType.entriesIdentifier),
            ir.Ref("i", TInt32())))))))
    assert(rTyp.isOfType(rowType))

    val newMatrixType = typ.copy(rvRowType = coerce[TStruct](rTyp))

    val keepF = { (mv: MatrixValue, keep: Array[Int]) =>
      val keepBc = mv.sparkContext.broadcast(keep)
      mv.copy(typ = newMatrixType,
        colValues = mv.colValues.copy(value = keep.map(mv.colValues.value)),
        rvd = mv.rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType, { (ctx, it) =>
          val f = makeF()
          val keep = keepBc.value
          var rv2 = RegionValue()

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

  def collectColsByKey(typ: MatrixType): (MatrixType, MatrixValue => MatrixValue) = {
    val oldRVRowType = typ.rvRowType
    val oldEntryArrayType = typ.entryArrayType
    val oldEntryType = typ.entryType

    val newColValueType = TStruct(typ.colValueStruct.fields.map(f => f.copy(typ = TArray(f.typ, required = true))))
    val newColType = typ.colKeyStruct ++ newColValueType
    val newEntryType = TStruct(typ.entryType.fields.map(f => f.copy(typ = TArray(f.typ, required = true))))
    val newMatrixType = typ.copyParts(colType = newColType, entryType = newEntryType)
    val newRVRowType = newMatrixType.rvRowType
    val localRowSize = newRVRowType.size

    (newMatrixType, { mv =>
      val colValMap: Map[Row, Array[Int]] = mv.colValues.value
        .map(_.asInstanceOf[Row])
        .zipWithIndex
        .groupBy[Row] { case (r, i) => typ.extractColKey(r) }
        .mapValues {
          _.map { case (r, i) => i }.toArray
        }
      val idxMap = colValMap.values.toArray

      val newColValues: BroadcastIndexedSeq = mv.colValues.copy(
        value = colValMap.map { case (key, idx) =>
          Row.fromSeq(key.toSeq ++ newColValueType.fields.map { f =>
            idx.map { i =>
              mv.colValues.value(i).asInstanceOf[Row]
                .get(typ.colValueFieldIdx(f.index))
            }.toIndexedSeq
          })
        }.toIndexedSeq)

      val newRVD = mv.rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType) { it =>
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        it.map { rv =>
          val entryArrayOffset = oldRVRowType.loadField(rv, oldRVRowType.fieldIdx(MatrixType.entriesIdentifier))

          rvb.set(rv.region)
          rvb.start(newRVRowType)
          rvb.startStruct()
          var i = 0
          while (i < localRowSize - 1) {
            rvb.addField(oldRVRowType, rv, i)
            i += 1
          }
          rvb.startArray(idxMap.length) // start entries array
          i = 0
          while (i < idxMap.length) {
            rvb.startStruct()
            var j = 0
            while (j < newEntryType.size) {
              rvb.startArray(idxMap(i).length)
              var k = 0
              while (k < idxMap(i).length) {
                rvb.addField(
                  oldEntryType,
                  rv.region,
                  oldEntryArrayType.loadElement(rv.region, entryArrayOffset, idxMap(i)(k)),
                  j
                )
                k += 1
              }
              rvb.endArray()
              j += 1
            }
            rvb.endStruct()
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv2.set(rv.region, rvb.end())
          rv2
        }
      }

      mv.copy(
        typ = newMatrixType,
        colValues = newColValues,
        rvd = newRVD
      )
    })
  }
}

abstract sealed class MatrixIR extends BaseIR {
  def typ: MatrixType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def execute(hc: HailContext): MatrixValue
}

case class MatrixLiteral(
  typ: MatrixType,
  value: MatrixValue) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def execute(hc: HailContext): MatrixValue = value

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixLiteral = {
    assert(newChildren.isEmpty)
    MatrixLiteral(typ, value)
  }

  override def toString: String = "MatrixLiteral(...)"
}

object MatrixReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[MatrixNativeReader], classOf[MatrixRangeReader]))
}

abstract class MatrixReader {
  def apply(mr: MatrixRead): MatrixValue
}

case class MatrixNativeReader(path: String, spec: MatrixTableSpec) extends MatrixReader {
  def apply(mr: MatrixRead): MatrixValue = {
    val hc = HailContext.get

    val requestedType = mr.typ
    assert(PruneDeadFields.isSupertype(requestedType, spec.matrix_type))

    val globals = spec.globalsComponent.readLocal(hc, path, requestedType.globalType)(0).asInstanceOf[Row]

    val colAnnotations =
      if (mr.dropCols)
        FastIndexedSeq.empty[Annotation]
      else
        spec.colsComponent.readLocal(hc, path, requestedType.colType).asInstanceOf[IndexedSeq[Annotation]]

    val rvd =
      if (mr.dropRows)
        OrderedRVD.empty(hc.sc, requestedType.orvdType)
      else {
        val fullRowType = requestedType.rvRowType
        val rowType = requestedType.rowType
        val localEntriesIndex = requestedType.entriesIdx

        val rowsRVD = spec.rowsComponent.read(hc, path, requestedType.rowType).asInstanceOf[OrderedRVD]
        if (mr.dropCols) {
          val (t2, makeF) = ir.Compile[Long, Long](
            "row", requestedType.rowType,
            MakeStruct(
              fullRowType.fields.zipWithIndex.map { case (f, i) =>
                  val v: IR = if (i == localEntriesIndex)
                    MakeArray(FastSeq.empty, TArray(requestedType.entryType))
                  else
                    GetField(Ref("row", requestedType.rowType), f.name)
                  f.name -> v
              }))
          assert(t2 == fullRowType)

          rowsRVD.mapPartitionsPreservesPartitioning(requestedType.orvdType) { it =>
            val f = makeF()
            val rv2 = RegionValue()
            it.map { rv =>
              val off = f(rv.region, rv.offset, false)
              rv2.set(rv.region, off)
              rv2
            }
          }
        } else {
          val entriesRVD = spec.entriesComponent.read(hc, path, requestedType.entriesRVType)
          val entriesRowType = entriesRVD.rowType

          val (t2, makeF) = ir.Compile[Long, Long, Long](
            "row", requestedType.rowType,
            "entriesRow", entriesRowType,
            MakeStruct(
              fullRowType.fields.zipWithIndex.map { case (f, i) =>
                val v: IR = if (i == localEntriesIndex)
                  GetField(Ref("entriesRow", entriesRowType), MatrixType.entriesIdentifier)
                else
                  GetField(Ref("row", requestedType.rowType), f.name)
                f.name -> v
              }))
          assert(t2 == fullRowType)

          rowsRVD.zipPartitions(requestedType.orvdType, rowsRVD.partitioner, entriesRVD, preservesPartitioning = true) { (ctx, it1, it2) =>
            val f = makeF()
            val rvb = ctx.rvb
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
  val typ: MatrixType = MatrixType.fromParts(
    globalType = TStruct.empty(),
    colKey = Array("col_idx"),
    colType = TStruct("col_idx" -> TInt32()),
    rowPartitionKey = Array("row_idx"),
    rowKey = Array("row_idx"),
    rowType = TStruct("row_idx" -> TInt32()),
    entryType = TStruct.empty())

  def apply(mr: MatrixRead): MatrixValue = {
    assert(mr.typ == typ)

    val partCounts = mr.partitionCounts.get.map(_.toInt)
    val nPartitionsAdj = mr.partitionCounts.get.length

    val hc = HailContext.get
    val localRVType = typ.rvRowType
    val partStarts = partCounts.scanLeft(0)(_ + _)
    val localNCols = if (mr.dropCols) 0 else nCols

    val rvd = if (mr.dropRows)
      OrderedRVD.empty(hc.sc, typ.orvdType)
    else {
      OrderedRVD(typ.orvdType,
        new OrderedRVDPartitioner(typ.rowPartitionKey.toArray,
          typ.rowKeyStruct,
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

    MatrixValue(typ,
      BroadcastRow(Row(), typ.globalType, hc.sc),
      BroadcastIndexedSeq(
        Iterator.range(0, localNCols)
          .map(Row(_))
          .toFastIndexedSeq,
        TArray(typ.colType),
        hc.sc),
      rvd)
  }
}

case class MatrixRead(
  typ: MatrixType,
  override val partitionCounts: Option[IndexedSeq[Long]],
  dropCols: Boolean,
  dropRows: Boolean,
  reader: MatrixReader) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRead = {
    assert(newChildren.isEmpty)
    MatrixRead(typ, partitionCounts, dropCols, dropRows, reader)
  }

  def execute(hc: HailContext): MatrixValue = reader(this)

  override def toString: String = s"MatrixRead($typ, partitionCounts = $partitionCounts, dropCols = $dropCols, dropRows = $dropRows)"
}

case class MatrixFilterCols(child: MatrixIR, pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterCols = {
    assert(newChildren.length == 2)
    MatrixFilterCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val (typ, filterF) = MatrixIR.filterCols(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobals = prev.globals.broadcast
    val localColType = typ.colType

    val (rTyp, predCompiledFunc) = ir.Compile[Long, Long, Boolean](
      "global", typ.globalType,
      "sa", typ.colType,
      pred)

    val p = (sa: Annotation, i: Int) => {
      Region.scoped { colRegion =>
        // FIXME: it would be nice to only load the globals once per matrix
        val rvb = new RegionValueBuilder(colRegion)
        rvb.start(typ.globalType)
        rvb.addAnnotation(typ.globalType, localGlobals.value)
        val globalRVoffset = rvb.end()

        val colRVb = new RegionValueBuilder(colRegion)
        colRVb.start(localColType)
        colRVb.addAnnotation(localColType, sa)
        val colRVoffset = colRVb.end()
        predCompiledFunc()(colRegion, globalRVoffset, false, colRVoffset, false)
      }
    }

    filterF(prev, p)
  }
}

case class MatrixFilterRows(child: MatrixIR, pred: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterRows = {
    assert(newChildren.length == 2)
    MatrixFilterRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  def typ: MatrixType = child.typ

  val tAggElt: Type = child.typ.entryType
  val aggSymTab = Map(
    "global" -> (0, child.typ.globalType),
    "va" -> (1, child.typ.rvRowType),
    "g" -> (2, child.typ.entryType),
    "sa" -> (3, child.typ.colType))

  val tAgg = TAggregable(tAggElt, aggSymTab)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(child.typ == prev.typ)

    val localGlobalsType = prev.typ.globalType
    val globalsBc = prev.globals.broadcast

    val vaType = prev.typ.rvRowType
    val (rTyp, f) = Compile[Long, Long, Boolean](
      "global", prev.typ.globalType,
      "va", vaType,
      pred)

    val filteredRDD = prev.rvd.mapPartitionsPreservesPartitioning(prev.typ.orvdType, { (ctx, it) =>
      val rvb = new RegionValueBuilder()
      val predicate = f()

      val partRegion = ctx.freshContext.region

      rvb.set(partRegion)
      rvb.start(localGlobalsType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val globals = rvb.end()

      it.filter { rv =>
        val region = rv.region
        val row = rv.offset

        if (predicate(region, globals, false, row, false))
          true
        else {
          ctx.region.clear()
          false
        }
      }
    })

    prev.copy(rvd = filteredRDD)
  }
}

case class ChooseCols(child: MatrixIR, oldIndices: Array[Int]) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): ChooseCols = {
    assert(newChildren.length == 1)
    ChooseCols(newChildren(0).asInstanceOf[MatrixIR], oldIndices)
  }

  val (typ, colsF) = MatrixIR.chooseColsWithArray(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    colsF(prev, oldIndices)
  }
}

case class CollectColsByKey(child: MatrixIR) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): CollectColsByKey = {
    assert(newChildren.length == 1)
    CollectColsByKey(newChildren(0).asInstanceOf[MatrixIR])
  }

  val (typ, groupF) = MatrixIR.collectColsByKey(child.typ)

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    groupF(prev)
  }
}

case class MatrixAggregateRowsByKey(child: MatrixIR, expr: IR) extends MatrixIR {
  require(child.typ.rowKey.nonEmpty)

  def children: IndexedSeq[BaseIR] = Array(child, expr)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateRowsByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: MatrixIR, newExpr: IR) = newChildren
    MatrixAggregateRowsByKey(newChild, newExpr)
  }

  val typ: MatrixType = child.typ.copyParts(
    rowType = child.typ.orvdType.kType,
    entryType = coerce[TStruct](expr.typ)
  )

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val nCols = prev.nCols

    val (rvAggs, makeInit, makeSeq, aggResultType, makeAnnotate, rTyp) = ir.CompileWithAggregators[Long, Long, Long, Long, Long](
      "global", child.typ.globalType,
      "global", child.typ.globalType,
      "colValues", TArray(child.typ.colType),
      "va", child.typ.rvRowType,
      expr, { (nAggs, initializeIR) =>
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
              ir.Recur(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(nCols), ir.I32(1)),
          colIdx,
          rewrite(initializeIR))
      }, { (nAggs, sequenceIR) =>
        val colIdx = ir.genUID()

        def rewrite(x: IR): IR = {
          x match {
            case SeqOp(a, i, agg, args) =>
              SeqOp(a,
                ir.ApplyBinaryPrimOp(ir.Add(),
                  ir.ApplyBinaryPrimOp(ir.Multiply(), ir.Ref(colIdx, TInt32()), ir.I32(nAggs)),
                  i),
                agg, args)
            case _ =>
              ir.Recur(rewrite)(x)
          }
        }

        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(nCols), ir.I32(1)),
          colIdx,
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", TArray(prev.typ.colType)), ir.Ref(colIdx, TInt32())),
            ir.Let("g", ir.ArrayRef(
              ir.GetField(ir.Ref("va", prev.typ.rvRowType), MatrixType.entriesIdentifier),
              ir.Ref(colIdx, TInt32())),
              rewrite(sequenceIR))))
      })
    val nAggs = rvAggs.length

    assert(coerce[TStruct](rTyp) == typ.entryType, s"$rTyp, ${ typ.entryType }")

    val newRVType = typ.rvRowType
    val newRowType = typ.rowType
    val rvType = prev.typ.rvRowType
    val selectIdx = prev.typ.orvdType.kRowFieldIdx
    val keyOrd = prev.typ.orvdType.kRowOrd
    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast
    val newRVD = prev.rvd.boundary.mapPartitionsPreservesPartitioning(typ.orvdType, { (ctx, it) =>
      val rvb = new RegionValueBuilder()
      val partRegion = ctx.freshContext.region

      rvb.set(partRegion)
      rvb.start(localGlobalsType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val globals = rvb.end()

      rvb.start(localColsType)
      rvb.addAnnotation(localColsType, colValuesBc.value)
      val cols = rvb.end()

      val initialize = makeInit()
      val sequence = makeSeq()
      val annotate = makeAnnotate()

      new Iterator[RegionValue] {
        var isEnd = false
        var current: RegionValue = _
        val rvRowKey: WritableRegionValue = WritableRegionValue(newRowType, ctx.freshRegion)
        val consumerRegion = ctx.region
        val newRV = RegionValue(consumerRegion)

        val colRVAggs = new Array[RegionValueAggregator](nAggs * nCols)

        {
          var i = 0
          while (i < nCols) {
            var j = 0
            while (j < nAggs) {
              colRVAggs(i * nAggs + j) = rvAggs(j).newInstance()
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

          rvRowKey.setSelect(rvType, selectIdx, current)

          colRVAggs.foreach(_.clear())

          initialize(current.region, colRVAggs, globals, false)

          do {
            sequence(current.region, colRVAggs,
              globals, false,
              cols, false,
              current.offset, false)
            current = null
          } while (hasNext && keyOrd.equiv(rvRowKey.value, current))

          rvb.set(consumerRegion)

          val aggResultsOffsets = Array.tabulate(nCols) { i =>
            rvb.start(aggResultType)
            rvb.startStruct()
            var j = 0
            while (j < nAggs) {
              colRVAggs(i * nAggs + j).result(rvb)
              j += 1
            }
            rvb.endStruct()
            rvb.end()
          }

          rvb.start(newRVType)
          rvb.startStruct()

          {
            var i = 0
            while (i < newRowType.size) {
              rvb.addField(newRowType, rvRowKey.value, i)
              i += 1
            }
          }

          rvb.startArray(nCols)

          {
            var i = 0
            while (i < nCols) {
              val newEntryOff = annotate(consumerRegion,
                aggResultsOffsets(i), false,
                globals, false)

              rvb.addRegionValue(rTyp, consumerRegion, newEntryOff)

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

case class MatrixAggregateColsByKey(child: MatrixIR, aggIR: IR) extends MatrixIR {
  require(child.typ.colKey.nonEmpty)

  def children: IndexedSeq[BaseIR] = Array(child, aggIR)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixAggregateColsByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: MatrixIR, newExpr: IR) = newChildren
    MatrixAggregateColsByKey(newChild, newExpr)
  }

  val typ = {
    val newEntryType = aggIR.typ
    child.typ.copyParts(entryType = coerce[TStruct](newEntryType), colType = child.typ.colKeyStruct)
  }

  def execute(hc: HailContext): MatrixValue = {
    val mv = child.execute(hc)

    // local things for serialization
    val oldNCols = mv.nCols
    val oldRVRowType = mv.typ.rvRowType
    val oldColsType = TArray(mv.typ.colType)
    val oldColValues = mv.colValues
    val oldColValuesBc = mv.colValues.broadcast
    val oldGlobalsBc = mv.globals.broadcast
    val oldGlobalsType = mv.typ.globalType

    val newRVType = typ.rvRowType
    val newColType = typ.colType
    val newEntriesIndex = typ.entriesIdx

    val keyIndices = mv.typ.colKey.map(k => mv.typ.colType.field(k).index)
    val keys = oldColValuesBc.value.map { a => Row.fromSeq(keyIndices.map(a.asInstanceOf[Row].get)) }.toSet.toArray
    val nKeys = keys.length
    val newColValues = oldColValues.copy(value = keys, t = TArray(newColType))

    val keysByColumn = oldColValues.value.map { sa => Row.fromSeq(keyIndices.map(sa.asInstanceOf[Row].get)) }
    val keyMap = keys.zipWithIndex.toMap
    val newColumnIndices = keysByColumn.map { k => keyMap(k) }.toArray
    val newColumnIndicesType = TArray(TInt32())

    val transformInitOp: (Int, IR) => IR = { (nAggs, initOpIR) =>
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case InitOp(i, args, aggSig) =>
            InitOp(ir.ApplyBinaryPrimOp(ir.Add(),
              ir.ApplyBinaryPrimOp(
                ir.Multiply(),
                ir.ArrayRef(ir.Ref("newColumnIndices", newColumnIndicesType), ir.Ref(colIdx, TInt32())),
                ir.I32(nAggs)),
              i),
              args,
              aggSig)
          case _ =>
            ir.Recur(rewrite)(x)
        }
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(oldNCols), ir.I32(1)),
        colIdx,
        rewrite(initOpIR))
    }

    val transformSeqOp: (Int, IR) => IR = { (nAggs, seqOpIR) =>
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case SeqOp(a, i, aggSig, args) =>
            SeqOp(a,
              ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(
                  ir.Multiply(),
                  ir.ArrayRef(ir.Ref("newColumnIndices", newColumnIndicesType), ir.Ref(colIdx, TInt32())),
                  ir.I32(nAggs)),
                i),
              aggSig, args)
          case _ =>
            ir.Recur(rewrite)(x)
        }
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(oldNCols), ir.I32(1)),
        colIdx,
        ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", oldColsType), ir.Ref(colIdx, TInt32())),
          ir.Let("g", ir.ArrayRef(
            ir.GetField(ir.Ref("va", oldRVRowType), MatrixType.entriesIdentifier),
            ir.Ref(colIdx, TInt32())),
            rewrite(seqOpIR)
          )))
    }

    val (rvAggs, initOps, seqOps, aggResultType, f, rTyp) = ir.CompileWithAggregators[Long, Long, Long, Long, Long, Long, Long, Long](
      "global", oldGlobalsType,
      "va", oldRVRowType,
      "newColumnIndices", newColumnIndicesType,
      "global", oldGlobalsType,
      "colValues", oldColsType,
      "va", oldRVRowType,
      "newColumnIndices", newColumnIndicesType,
      aggIR,
      transformInitOp,
      transformSeqOp)
    assert(rTyp == typ.entryType)

    val nAggs = rvAggs.length

    val colRVAggs = new Array[RegionValueAggregator](nAggs * nKeys)
    var i = 0
    while (i < nKeys) {
      var j = 0
      while (j < nAggs) {
        colRVAggs(i * nAggs + j) = rvAggs(j).newInstance()
        j += 1
      }
      i += 1
    }

    val mapPartitionF = { (ctx: RVDContext, it: Iterator[RegionValue]) =>
      val rvb = new RegionValueBuilder()
      val newRV = RegionValue()

      val partitionRegion = ctx.freshContext.region

      rvb.set(partitionRegion)
      rvb.start(oldGlobalsType)
      rvb.addAnnotation(oldGlobalsType, oldGlobalsBc.value)
      val partitionWideGlobalsOffset = rvb.end()

      rvb.start(oldColsType)
      rvb.addAnnotation(oldColsType, oldColValuesBc.value)
      val partitionWideColumnsOffset = rvb.end()

      rvb.start(newColumnIndicesType)
      rvb.startArray(newColumnIndices.length)
      var i = 0
      while (i < newColumnIndices.length) {
        rvb.addInt(newColumnIndices(i))
        i += 1
      }
      rvb.endArray()
      val partitionWideMapOffset = rvb.end()

      it.map { rv =>
        val oldRow = rv.offset

        rvb.set(rv.region)
        rvb.start(oldGlobalsType)
        rvb.addRegionValue(oldGlobalsType, partitionRegion, partitionWideGlobalsOffset)
        val globalsOffset = rvb.end()

        rvb.set(rv.region)
        rvb.start(oldColsType)
        rvb.addRegionValue(oldColsType, partitionRegion, partitionWideColumnsOffset)
        val columnsOffset = rvb.end()

        rvb.set(rv.region)
        rvb.start(newColumnIndicesType)
        rvb.addRegionValue(newColumnIndicesType, partitionRegion, partitionWideMapOffset)
        val mapOffset = rvb.end()

        var j = 0
        while (j < colRVAggs.length) {
          colRVAggs(j).clear()
          j += 1
        }

        initOps()(rv.region, colRVAggs, globalsOffset, false, oldRow, false, mapOffset, false)
        seqOps()(rv.region, colRVAggs, globalsOffset, false, columnsOffset, false, oldRow, false, mapOffset, false)

        val resultOffsets = Array.tabulate(nKeys) { i =>
          var j = 0
          rvb.start(aggResultType)
          rvb.startStruct()
          while (j < nAggs) {
            colRVAggs(i * nAggs + j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val aggResultOffset = rvb.end()
          f()(rv.region, aggResultOffset, false, globalsOffset, false, oldRow, false, mapOffset, false)
        }

        rvb.start(newRVType)
        rvb.startStruct()
        var k = 0
        while (k < newEntriesIndex) {
          rvb.addField(oldRVRowType, rv, k)
          k += 1
        }

        i = 0
        rvb.startArray(nKeys)
        while (i < nKeys) {
          rvb.addRegionValue(rTyp, rv.region, resultOffsets(i))
          i += 1
        }
        rvb.endArray()
        k += 1

        while (k < newRVType.fields.length) {
          rvb.addField(oldRVRowType, rv, k)
          k += 1
        }

        rvb.endStruct()
        rv.setOffset(rvb.end())
        rv
      }
    }

    val newRVD = mv.rvd.mapPartitionsPreservesPartitioning(typ.orvdType, mapPartitionF)
    mv.copy(typ = typ, colValues = newColValues, rvd = newRVD)
  }
}

case class MatrixMapEntries(child: MatrixIR, newEntries: IR) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, newEntries)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapEntries = {
    assert(newChildren.length == 2)
    MatrixMapEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val newRow = {
    val vaType = child.typ.rvRowType
    val saType = TArray(child.typ.colType)

    val arrayLength = ArrayLen(GetField(Ref("va", vaType), MatrixType.entriesIdentifier))
    val idxEnv = new Env[IR]()
      .bind("g", ArrayRef(GetField(Ref("va", vaType), MatrixType.entriesIdentifier), Ref("i", TInt32())))
      .bind("sa", ArrayRef(Ref("sa", saType), Ref("i", TInt32())))
    val entries = ArrayMap(ArrayRange(I32(0), arrayLength, I32(1)), "i", Subst(newEntries, idxEnv))
    InsertFields(Ref("va", vaType), Seq((MatrixType.entriesIdentifier, entries)))
  }

  val typ: MatrixType =
    child.typ.copy(rvRowType = newRow.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val localGlobalsType = typ.globalType
    val localColsType = TArray(typ.colType)
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long](
      "global", localGlobalsType,
      "va", prev.typ.rvRowType,
      "sa", localColsType,
      newRow)
    assert(rTyp == typ.rvRowType)

    val newRVD = prev.rvd.mapPartitionsPreservesPartitioning(typ.orvdType, { (ctx, it) =>
      val rvb = new RegionValueBuilder()
      val newRV = RegionValue()
      val rowF = f()
      val partitionRegion = ctx.freshRegion

      rvb.set(partitionRegion)
      rvb.start(localGlobalsType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val globals = rvb.end()

      rvb.start(localColsType)
      rvb.addAnnotation(localColsType, colValuesBc.value)
      val cols = rvb.end()

      it.map { rv =>
        val region = rv.region
        val oldRow = rv.offset

        val off = rowF(region, globals, false, oldRow, false, cols, false)

        newRV.set(region, off)
        newRV
      }
    })
    prev.copy(typ = typ, rvd = newRVD)
  }
}

case class MatrixMapRows(child: MatrixIR, newRow: IR, newKey: Option[(IndexedSeq[String], IndexedSeq[String])]) extends MatrixIR {

  def children: IndexedSeq[BaseIR] = Array(child, newRow)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapRows = {
    assert(newChildren.length == 2)
    MatrixMapRows(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], newKey)
  }

  val newRVRow = InsertFields(newRow, Seq(
    MatrixType.entriesIdentifier -> GetField(Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier)))

  val typ: MatrixType = {
    val newRowKey = newKey.map { case (pk, k) => pk ++ k }.getOrElse(child.typ.rowKey)
    val newPartitionKey = newKey.map { case (pk, _) => pk }.getOrElse(child.typ.rowPartitionKey)
    child.typ.copy(rvRowType = newRVRow.typ, rowKey = newRowKey, rowPartitionKey = newPartitionKey)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(prev.typ == child.typ)

    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val localNCols = prev.nCols
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val colValuesType = TArray(prev.typ.colType)
    val vaType = prev.typ.rvRowType
    val (rvAggs, initOps, seqOps, aggResultType, f, rTyp) = ir.CompileWithAggregators[Long, Long, Long, Long, Long, Long](
      "global", prev.typ.globalType,
      "va", vaType,
      "global", prev.typ.globalType,
      "colValues", colValuesType,
      "va", vaType,
      newRVRow,
      (nAggs: Int, initOpIR: IR) => initOpIR, { (nAggs: Int, seqOpIR: IR) =>
        ir.ArrayFor(
          ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
          "i",
          ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref("i", TInt32())),
            ir.Let("g", ir.ArrayRef(
              ir.GetField(ir.Ref("va", vaType), MatrixType.entriesIdentifier),
              ir.Ref("i", TInt32())),
              seqOpIR)))
      })
    assert(rTyp == typ.rvRowType, s"$rTyp, ${ typ.rvRowType }")

    val mapPartitionF = { (ctx: RVDContext, it: Iterator[RegionValue]) =>
      val rvb = new RegionValueBuilder()
      val newRV = RegionValue()
      val rowF = f()

      val partRegion = ctx.freshContext.region

      rvb.set(partRegion)
      rvb.start(localGlobalsType)
      rvb.addAnnotation(localGlobalsType, globalsBc.value)
      val globals = rvb.end()

      val cols = if (rvAggs.nonEmpty) {
        rvb.start(localColsType)
        rvb.addAnnotation(localColsType, colValuesBc.value)
        rvb.end()
      } else 0L

      it.map { rv =>
        val region = rv.region
        val oldRow = rv.offset

        val aggResultsOff = if (rvAggs.nonEmpty) {
          var j = 0
          while (j < rvAggs.length) {
            rvAggs(j).clear()
            j += 1
          }

          initOps()(region, rvAggs, globals, false, oldRow, false)
          seqOps()(region, rvAggs, globals, false, cols, false, oldRow, false)

          rvb.start(aggResultType)
          rvb.startStruct()

          j = 0
          while (j < rvAggs.length) {
            rvAggs(j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val aggResultsOff = rvb.end()
          aggResultsOff
        } else
          0

        val off = rowF(region, aggResultsOff, false, globals, false, oldRow, false)

        newRV.set(region, off)
        newRV
      }
    }

    if (newKey.isDefined) {
      prev.copy(typ = typ,
        rvd = OrderedRVD.coerce(
          typ.orvdType,
          prev.rvd.mapPartitions(typ.rvRowType, mapPartitionF)))
    } else {
      val newRVD = prev.rvd.mapPartitionsPreservesPartitioning(typ.orvdType, mapPartitionF)
      prev.copy(typ = typ, rvd = newRVD)
    }
  }
}

case class MatrixMapCols(child: MatrixIR, newCol: IR, newKey: Option[IndexedSeq[String]]) extends MatrixIR {
  def children: IndexedSeq[BaseIR] = Array(child, newCol)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapCols = {
    assert(newChildren.length == 2)
    MatrixMapCols(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], newKey)
  }

  val tAggElt: Type = child.typ.entryType
  val aggSymTab = Map(
    "global" -> (0, child.typ.globalType),
    "va" -> (1, child.typ.rvRowType),
    "g" -> (2, child.typ.entryType),
    "sa" -> (3, child.typ.colType))

  val tAgg = TAggregable(tAggElt, aggSymTab)

  val typ: MatrixType = {
    val newColType = newCol.typ.asInstanceOf[TStruct]
    val newColKey = newKey.getOrElse(child.typ.colKey)
    child.typ.copy(colKey = newColKey, colType = newColType)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)
    assert(prev.typ == child.typ)

    val localGlobalsType = prev.typ.globalType
    val localColsType = TArray(prev.typ.colType)
    val localNCols = prev.nCols
    val colValuesBc = prev.colValues.broadcast
    val globalsBc = prev.globals.broadcast

    val colValuesType = TArray(prev.typ.colType)
    val vaType = prev.typ.rvRowType

    val (rvAggs,
    (preInitOp, initOpCompiler),
    (preSeqOp, seqOpCompiler),
    aggResultType,
    (postAgg, postAggCompiler)
      ) = ir.CompileWithAggregators.toIRs[
      Long, Long, Long, Long, Long, Long
      ]("global", localGlobalsType,
      "sa", prev.typ.colType,
      "global", localGlobalsType,
      "colValues", colValuesType,
      "va", vaType,
      newCol)

    val nAggs = rvAggs.length
    val initOpNeedsSA = Mentions(preInitOp, "sa")
    val initOpNeedsGlobals = Mentions(preInitOp, "global")
    val seqOpNeedsSA = Mentions(preSeqOp, "sa")
    val seqOpNeedsGlobals = Mentions(preSeqOp, "global")

    val (_, initOps) = initOpCompiler({
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
            ir.Recur(rewrite)(x)
        }
      }

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
        colIdx,
        rewrite(preInitOp))
    })

    val (_, seqOps) = seqOpCompiler({
      val colIdx = ir.genUID()

      def rewrite(x: IR): IR = {
        x match {
          case SeqOp(a, i, aggSig, args) =>
            SeqOp(a,
              ir.ApplyBinaryPrimOp(ir.Add(),
                ir.ApplyBinaryPrimOp(ir.Multiply(), ir.Ref(colIdx, TInt32()), ir.I32(nAggs)),
                i),
              aggSig, args)
          case _ =>
            ir.Recur(rewrite)(x)
        }
      }

      var oneSampleSeqOp = ir.Let("g", ir.ArrayRef(
        ir.GetField(ir.Ref("va", vaType), MatrixType.entriesIdentifier),
        ir.Ref(colIdx, TInt32())),
        rewrite(preSeqOp)
      )

      if (seqOpNeedsSA)
        oneSampleSeqOp = ir.Let(
          "sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref(colIdx, TInt32())),
          oneSampleSeqOp)

      ir.ArrayFor(
        ir.ArrayRange(ir.I32(0), ir.I32(localNCols), ir.I32(1)),
        colIdx,
        oneSampleSeqOp)
    })

    val (rTyp, f) = postAggCompiler(postAgg)

    assert(rTyp == typ.colType, s"$rTyp, ${ typ.colType }")

    log.info(s"""MatrixMapCols: initOp ${ initOpNeedsGlobals } ${ initOpNeedsSA };
                |seqOp ${ seqOpNeedsGlobals } ${ seqOpNeedsSA }""".stripMargin)

    val depth = treeAggDepth(hc, prev.nPartitions)

    val colRVAggs = new Array[RegionValueAggregator](nAggs * localNCols)
    var i = 0
    while (i < localNCols) {
      var j = 0
      while (j < nAggs) {
        colRVAggs(i * nAggs + j) = rvAggs(j).newInstance()
        j += 1
      }
      i += 1
    }

    val aggResults = if (nAggs > 0) {
      Region.scoped { region =>
        val rvb: RegionValueBuilder = new RegionValueBuilder()
        rvb.set(region)

        val globals = if (initOpNeedsGlobals) {
          rvb.start(localGlobalsType)
          rvb.addAnnotation(localGlobalsType, globalsBc.value)
          rvb.end()
        } else 0L

        val cols = if (initOpNeedsSA) {
          rvb.start(localColsType)
          rvb.addAnnotation(localColsType, colValuesBc.value)
          rvb.end()
        } else 0L

        initOps()(region, colRVAggs, globals, false, cols, false)
      }

      prev.rvd.treeAggregate[Array[RegionValueAggregator]](colRVAggs)({ (colRVAggs, rv) =>
        val rvb = new RegionValueBuilder()
        val region = rv.region
        val oldRow = rv.offset

        val globals = if (seqOpNeedsGlobals) {
          rvb.set(region)
          rvb.start(localGlobalsType)
          rvb.addAnnotation(localGlobalsType, globalsBc.value)
          rvb.end()
        } else 0L

        val cols = if (seqOpNeedsSA) {
          rvb.start(localColsType)
          rvb.addAnnotation(localColsType, colValuesBc.value)
          rvb.end()
        } else 0L

        seqOps()(region, colRVAggs, globals, false, cols, false, oldRow, false)

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

    val mapF = (a: Annotation, i: Int) => {
      Region.scoped { region =>
        rvb.set(region)

        rvb.start(aggResultType)
        rvb.startStruct()
        var j = 0
        while (j < nAggs) {
          aggResults(i * nAggs + j).result(rvb)
          j += 1
        }
        rvb.endStruct()
        val aggResultsOffset = rvb.end()

        rvb.start(localGlobalsType)
        rvb.addAnnotation(localGlobalsType, globalsBc.value)
        val globalRVoffset = rvb.end()

        val colRVb = new RegionValueBuilder(region)
        colRVb.start(prevColType)
        colRVb.addAnnotation(prevColType, a)
        val colRVoffset = colRVb.end()

        val resultOffset = f()(region, aggResultsOffset, false, globalRVoffset, false, colRVoffset, false)

        SafeRow(coerce[TStruct](rTyp), region, resultOffset)
      }
    }

    val newColValues = BroadcastIndexedSeq(colValuesBc.value.zipWithIndex.map { case (a, i) => mapF(a, i) }, TArray(typ.colType), hc.sc)
    prev.copy(typ = typ, colValues = newColValues)
  }
}

case class MatrixMapGlobals(child: MatrixIR, newRow: IR, value: BroadcastRow) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: MatrixType =
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixMapGlobals = {
    assert(newChildren.length == 2)
    MatrixMapGlobals(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR], value)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): MatrixValue = {
    val prev = child.execute(hc)

    val (rTyp, f) = ir.Compile[Long, Long, Long](
      "global", child.typ.globalType,
      "value", value.t,
      newRow)
    assert(rTyp == typ.globalType)

    val newGlobals = Region.scoped { globalRegion =>
      val globalOff = prev.globals.toRegion(globalRegion)
      val valueOff = value.toRegion(globalRegion)
      val newOff = f()(globalRegion, globalOff, false, valueOff, false)

      prev.globals.copy(
        value = SafeRow(rTyp.asInstanceOf[TStruct], globalRegion, newOff),
        t = rTyp.asInstanceOf[TStruct])
    }

    prev.copy(typ = typ, globals = newGlobals)
  }
}

case class MatrixFilterEntries(child: MatrixIR, pred: IR) extends MatrixIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixFilterEntries = {
    assert(newChildren.length == 2)
    MatrixFilterEntries(newChildren(0).asInstanceOf[MatrixIR], newChildren(1).asInstanceOf[IR])
  }

  val typ: MatrixType = child.typ

  def execute(hc: HailContext): MatrixValue = {
    val mv = child.execute(hc)

    val localGlobalType = child.typ.globalType
    val globalsBc = mv.globals.broadcast
    val localColValuesType = TArray(child.typ.colType)
    val colValuesBc = mv.colValues.broadcast

    val colValuesType = TArray(child.typ.colType)

    val x = ir.InsertFields(ir.Ref("va", child.typ.rvRowType),
      FastSeq(MatrixType.entriesIdentifier ->
        ir.ArrayMap(ir.ArrayRange(ir.I32(0), ir.I32(mv.nCols), ir.I32(1)),
          "i",
          ir.Let("g",
            ir.ArrayRef(
              ir.GetField(ir.Ref("va", child.typ.rvRowType), MatrixType.entriesIdentifier),
              ir.Ref("i", TInt32())),
            ir.If(
              ir.Let("sa", ir.ArrayRef(ir.Ref("colValues", colValuesType), ir.Ref("i", TInt32())),
                pred),
              ir.Ref("g", child.typ.entryType),
              ir.NA(child.typ.entryType))))))

    val (t, f) = ir.Compile[Long, Long, Long, Long](
      "global", child.typ.globalType,
      "colValues", colValuesType,
      "va", child.typ.rvRowType,
      x)
    assert(t == typ.rvRowType)

    val mapPartitionF = { (ctx: RVDContext, it: Iterator[RegionValue]) =>
      val rvb = new RegionValueBuilder(ctx.freshRegion)
      rvb.start(localGlobalType)
      rvb.addAnnotation(localGlobalType, globalsBc.value)
      val globals = rvb.end()

      rvb.start(localColValuesType)
      rvb.addAnnotation(localColValuesType, colValuesBc.value)
      val cols = rvb.end()
      val rowF = f()

      val newRV = RegionValue()
      it.map { rv =>
        val off = rowF(rv.region, globals, false, cols, false, rv.offset, false)
        newRV.set(rv.region, off)
        newRV
      }
    }

    val newRVD = mv.rvd.mapPartitionsPreservesPartitioning(typ.orvdType, mapPartitionF)
    mv.copy(rvd = newRVD)
  }
}
