package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.types._
import is.hail.expr.{TableAnnotationImpex, ir}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.table.{Ascending, SortField, TableSpec}
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

object TableIR {
  def read(hc: HailContext, path: String, dropRows: Boolean = false, requestedType: Option[TableType]): TableIR = {
    val successFile = path + "/_SUCCESS"
    if (!hc.hadoopConf.exists(path + "/_SUCCESS"))
      fatal(s"write failed: file not found: $successFile")

    val spec = (RelationalSpec.read(hc, path): @unchecked) match {
      case ts: TableSpec => ts
      case _: MatrixTableSpec => fatal(s"file is a MatrixTable, not a Table: '$path'")
    }

    TableRead(path, spec, requestedType.getOrElse(spec.table_type), dropRows = false)
  }
}

abstract sealed class TableIR extends BaseIR {
  def typ: TableType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  def execute(hc: HailContext): TableValue
}

case class TableLiteral(value: TableValue) extends TableIR {
  val typ: TableType = value.typ

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    TableLiteral(value)
  }

  def execute(hc: HailContext): TableValue = value
}

case class TableRead(path: String, spec: TableSpec, typ: TableType, dropRows: Boolean) extends TableIR {
  assert(PruneDeadFields.isSupertype(typ, spec.table_type),
    s"\n  original:  ${ spec.table_type }\n  requested: $typ")

  override def partitionCounts: Option[IndexedSeq[Long]] = Some(spec.partitionCounts)

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(path, spec, typ, dropRows)
  }

  def execute(hc: HailContext): TableValue = {
    val globals = spec.globalsComponent.readLocal(hc, path, typ.globalType)(0)
    val rvd = if (dropRows)
      OrderedRVD.empty(hc.sc, typ.rvdType)
    else {
      val rvd = spec.rowsComponent.read(hc, path, typ.rowType)
      (rvd, typ.key) match {
        case (ordered: OrderedRVD, Some(k))
          if ordered.typ.key.startsWith(k) => ordered
        case (_, Some(k)) => rvd.toOrderedRVD.changeKey(k)
        case (_, None) => rvd.toOrderedRVD
      }
    }
    TableValue(
      typ,
      BroadcastRow(globals, typ.globalType, hc.sc),
      rvd.toOldStyleRVD)
  }
}

case class TableParallelize(rows: IR, nPartitions: Option[Int] = None) extends TableIR {
  require(rows.typ.isInstanceOf[TArray] && rows.typ.asInstanceOf[TArray].elementType.isInstanceOf[TStruct])

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(rows)

  def copy(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    val IndexedSeq(newRows: IR) = newChildren
    TableParallelize(newRows, nPartitions)
  }

  val typ: TableType = TableType(
    rows.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct],
    None,
    TStruct())

  def execute(hc: HailContext): TableValue = {
    val rowsValue = Interpret[IndexedSeq[Row]](rows, optimize = false)
    rowsValue.zipWithIndex.foreach { case (r, idx) =>
      if (r == null)
        fatal(s"cannot parallelize null values: found null value at index $idx")
    }

    log.info(s"parallelized ${ rowsValue.length } rows")

    val rowTyp = typ.rowType
    val rvd = ContextRDD.parallelize[RVDContext](hc.sc, rowsValue, nPartitions)
      .cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowTyp))
    TableValue(typ, BroadcastRow(Row(), typ.globalType, hc.sc), UnpartitionedRVD(rowTyp, rvd))
  }
}

case class TableImport(paths: Array[String], typ: TableType, readerOpts: TableReaderOptions) extends TableIR {
  assert(typ.key.isEmpty)
  assert(typ.globalType.size == 0)
  assert(typ.rowType.size == readerOpts.useColIndices.length)

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableImport = {
    assert(newChildren.isEmpty)
    TableImport(paths, typ, readerOpts)
  }

  def execute(hc: HailContext): TableValue = {
    val rowTyp = typ.rowType
    val nFieldOrig = readerOpts.originalType.size
    val rowFields = rowTyp.fields

    val useColIndices = readerOpts.useColIndices


    val rvd = ContextRDD.textFilesLines[RVDContext](hc.sc, paths, readerOpts.nPartitions)
      .filter { line =>
        !readerOpts.isComment(line.value) &&
          (readerOpts.noHeader || readerOpts.header != line.value) &&
          !(readerOpts.skipBlankLines && line.value.isEmpty)
      }.cmapPartitions { (ctx, it) =>
      val region = ctx.region
      val rvb = ctx.rvb
      val rv = RegionValue(region)

      val ab = new ArrayBuilder[String]
      val sb = new StringBuilder
      it.map {
        _.map { line =>
          val sp = TextTableReader.splitLine(line, readerOpts.separator, readerOpts.quote, ab, sb)
          if (sp.length != nFieldOrig)
            fatal(s"expected $nFieldOrig fields, but found ${ sp.length } fields")

          rvb.set(region)
          rvb.start(rowTyp)
          rvb.startStruct()

          var i = 0
          while (i < useColIndices.length) {
            val f = rowFields(i)
            val name = f.name
            val typ = f.typ
            val field = sp(useColIndices(i))
            try {
              if (field == readerOpts.missing)
                rvb.setMissing()
              else
                rvb.addAnnotation(typ, TableAnnotationImpex.importAnnotation(field, typ))
            } catch {
              case e: Exception =>
                fatal(s"""${ e.getClass.getName }: could not convert "$field" to $typ in column "$name" """, e)
            }
            i += 1
          }

          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }.value
      }
    }

    TableValue(typ, BroadcastRow(Row.empty, typ.globalType, hc.sc), UnpartitionedRVD(rowTyp, rvd))
  }
}

/**
  * Change the table to have key 'keys'.
  *
  * Let n be the longest common prefix of 'keys' and the old key, i.e. the
  * number of key fields that are not being changed.
  * - If 'isSorted', then 'child' must already be sorted by 'keys', and n must
  *   not be zero. Thus, if 'isSorted', TableKeyBy will not shuffle or scan. 
  *   The new partitioner will be the old one with partition bounds truncated
  *   to length n.
  * - If n = 'keys.length', i.e. we are simply shortening the key, do nothing
  *   but change the table type to the new key. 'isSorted' is ignored.
  * - Otherwise, if 'isSorted' is false and n < 'keys.length', then shuffle.
  */
case class TableKeyBy(child: TableIR, keys: IndexedSeq[String], isSorted: Boolean = false) extends TableIR {
  private val fields = child.typ.rowType.fieldNames.toSet
  assert(keys.forall(fields.contains), s"${ keys.filter(k => !fields.contains(k)).mkString(", ") }")

  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = if (keys.isEmpty) None else Some(keys))

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, isSorted)
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val orvd = tv.rvd.toOrderedRVD
    val nPreservedFields = keys.zip(orvd.typ.key).takeWhile { case (l, r) => l == r }.length
    assert(!isSorted || nPreservedFields > 0 || keys.isEmpty)

    val rvd = if (nPreservedFields == keys.length) {
      orvd
    } else if (isSorted) {
      orvd.truncateKey(keys.take(nPreservedFields))
        .extendKeyPreservesPartitioning(keys)
    } else {
      orvd.changeKey(keys)
    }.toOldStyleRVD
    tv.copy(typ = typ, rvd = rvd)
  }
}

case class TableRange(n: Int, nPartitions: Int) extends TableIR {
  require(n >= 0)
  require(nPartitions > 0)
  private val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRange = {
    assert(newChildren.isEmpty)
    TableRange(n, nPartitions)
  }

  private val partCounts = partition(n, nPartitionsAdj)

  override val partitionCounts = Some(partCounts.map(_.toLong).toFastIndexedSeq)

  val typ: TableType = TableType(
    TStruct("idx" -> TInt32()),
    Some(Array("idx")),
    TStruct.empty())

  def execute(hc: HailContext): TableValue = {
    val localRowType = typ.rowType
    val localPartCounts = partCounts
    val partStarts = partCounts.scanLeft(0)(_ + _)

    TableValue(typ,
      BroadcastRow(Row(), typ.globalType, hc.sc),
      new OrderedRVD(
        new OrderedRVDType(Array("idx"), typ.rowType),
        new OrderedRVDPartitioner(Array("idx"), typ.rowType,
          Array.tabulate(nPartitionsAdj) { i =>
            val start = partStarts(i)
            val end = partStarts(i + 1)
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          }),
        ContextRDD.parallelize(hc.sc, Range(0, nPartitionsAdj), nPartitionsAdj)
          .cmapPartitionsWithIndex { case (i, ctx, _) =>
            val region = ctx.region
            val rvb = ctx.rvb
            val rv = RegionValue(region)

            val start = partStarts(i)
            Iterator.range(start, start + localPartCounts(i))
              .map { j =>
                rvb.start(localRowType)
                rvb.startStruct()
                rvb.addInt(j)
                rvb.endStruct()
                rv.setOffset(rvb.end())
                rv
              }
          }))
  }
}

case class TableFilter(child: TableIR, pred: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  val typ: TableType = child.typ

  def copy(newChildren: IndexedSeq[BaseIR]): TableFilter = {
    assert(newChildren.length == 2)
    TableFilter(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)

    if (pred == True())
      return tv
    else if (pred == False())
      return tv.copy(rvd = OrderedRVD.empty(hc.sc, typ.rvdType).toOldStyleRVD)

    val (rTyp, f) = ir.Compile[Long, Long, Boolean](
      "row", child.typ.rowType,
      "global", child.typ.globalType,
      pred)
    assert(rTyp == TBoolean())

    tv.filterWithPartitionOp(f)((rowF, rv, globalRV) => rowF(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableHead(child: TableIR, n: Long) extends TableIR {
  require(n >= 0, fatal(s"TableHead: n must be non-negative! Found '$n'."))
  def typ: TableType = child.typ

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableHead(newChild, n)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] =
    child.partitionCounts.map(getHeadPartitionCounts(_, n))

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    prev.copy(rvd = prev.rvd.head(n, child.partitionCounts))
  }
}

case class TableRepartition(child: TableIR, n: Int, shuffle: Boolean) extends TableIR {
  def typ: TableType = child.typ

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRepartition(newChild, n, shuffle)
  }

  override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    prev.copy(rvd = prev.rvd.coalesce(n, shuffle))
  }
}

object TableJoin {
  def apply(left: TableIR, right: TableIR, joinType: String): TableJoin =
    TableJoin(left, right, joinType, left.typ.keyOrEmpty.length)
}

/**
  * Suppose 'left' has key [l_1, ..., l_n] and 'right' has key [r_1, ..., r_m].
  * Then [l_1, ..., l_j] and [r_1, ..., r_j] must have the same type, where
  * j = 'joinKey'. TableJoin computes the join of 'left' and 'right' along this
  * common prefix of their keys, returning a table with key
  * [l_1, ..., l_j, l_{j+1}, ..., l_n, r_{j+1}, ..., r_m] (with possible
  * renaming of right field names to avoid collision).
  *
  * WARNING: If 'left' has any duplicate (full) key [k_1, ..., k_n], and j < m,
  * and 'right' has multiple rows with the corresponding join key
  * [k_1, ..., k_j] but distinct full keys, then the resulting table will have
  * out-of-order keys. To avoid this, ensure one of the following:
  *   * j == m
  *   * 'left' has distinct keys
  *   * 'right' has distinct join keys (length j prefix), or at least no
  *     distinct keys with the same join key.
  */
case class TableJoin(left: TableIR, right: TableIR, joinType: String, joinKey: Int)
  extends TableIR {

  require(joinKey >= 0)
  require(left.typ.keyType.zip(right.typ.keyType).exists { case (leftKey, rightKey) =>
    (leftKey.size >= joinKey) && (rightKey.size >= joinKey) &&
      (leftKey.truncate(joinKey) isIsomorphicTo rightKey.truncate(joinKey))
  })

  val children: IndexedSeq[BaseIR] = Array(left, right)

  private val leftRVDType =
    OrderedRVDType(left.typ.key.get.take(joinKey), left.typ.rowType)
  private val rightRVDType =
    OrderedRVDType(right.typ.key.get.take(joinKey), right.typ.rowType)

  private val joinedFields =
    leftRVDType.kType.fields ++
      leftRVDType.valueType.fields ++
      rightRVDType.valueType.fields
  private val preNames = joinedFields.map(_.name).toArray
  private val (finalColumnNames, remapped) = mangle(preNames)

  private val joinedGlobalFields = left.typ.globalType.fields ++ right.typ.globalType.fields
  private val (finalGlobalNames, _) = mangle(joinedGlobalFields.map(_.name).toArray)
  val newGlobalType = TStruct(joinedGlobalFields.zipWithIndex.map {
    case (field, i) => (finalGlobalNames(i), field.typ)
  }: _*)

  val rightFieldMapping: Map[String, String] = {
    val remapMap = remapped.toMap
    (rightRVDType.key.iterator.zip(leftRVDType.key.iterator) ++
      rightRVDType.valueType.fieldNames.iterator.map(f => f -> remapMap.getOrElse(f, f))).toMap
  }

  val newRowType = TStruct(joinedFields.zipWithIndex.map {
    case (field, i) => (finalColumnNames(i), field.typ)
  }: _*)

  val newKey = left.typ.key.get ++ right.typ.key.get.drop(joinKey).map(rightFieldMapping)

  val typ: TableType = TableType(newRowType, Some(newKey), newGlobalType)

  def copy(newChildren: IndexedSeq[BaseIR]): TableJoin = {
    assert(newChildren.length == 2)
    TableJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      joinType,
      joinKey)
  }

  private val rvMerger = {
    val leftRowType = leftRVDType.rowType
    val rightRowType = rightRVDType.rowType
    val leftKeyFieldIdx = leftRVDType.kFieldIdx
    val rightKeyFieldIdx = rightRVDType.kFieldIdx
    val leftValueFieldIdx = leftRVDType.valueFieldIdx
    val rightValueFieldIdx = rightRVDType.valueFieldIdx
    val localNewRowType = newRowType

    { (_: RVDContext, it: Iterator[JoinedRegionValue]) =>
      val rvb = new RegionValueBuilder()
      val rv = RegionValue()
      it.map { joined =>
        val lrv = joined._1
        val rrv = joined._2

        if (lrv != null)
          rvb.set(lrv.region)
        else {
          assert(rrv != null)
          rvb.set(rrv.region)
        }

        rvb.start(localNewRowType)
        rvb.startStruct()

        if (lrv != null)
          rvb.addFields(leftRowType, lrv, leftKeyFieldIdx)
        else {
          assert(rrv != null)
          rvb.addFields(rightRowType, rrv, rightKeyFieldIdx)
        }

        if (lrv != null)
          rvb.addFields(leftRowType, lrv, leftValueFieldIdx)
        else
          rvb.skipFields(leftValueFieldIdx.length)

        if (rrv != null)
          rvb.addFields(rightRowType, rrv, rightValueFieldIdx)
        else
          rvb.skipFields(rightValueFieldIdx.length)

        rvb.endStruct()
        rv.set(rvb.region, rvb.end())
        rv
      }
    }
  }

  def execute(hc: HailContext): TableValue = {
    val leftTV = left.execute(hc)
    val rightTV = right.execute(hc)

    val newGlobals = BroadcastRow(
      Row.merge(leftTV.globals.value, rightTV.globals.value),
      newGlobalType,
      leftTV.rvd.sparkContext)

    val leftORVD = leftTV.rvd.asInstanceOf[OrderedRVD]
    val rightORVD = rightTV.rvd.asInstanceOf[OrderedRVD]
    val joinedRVD = leftORVD.orderedJoin(
      rightORVD,
      joinKey,
      joinType,
      rvMerger,
      new OrderedRVDType(newKey, newRowType))

    TableValue(typ, newGlobals, joinedRVD)
  }
}

case class TableLeftJoinRightDistinct(left: TableIR, right: TableIR, root: String) extends TableIR {
  require(left.typ.keyType.exists(l => right.typ.keyType.exists(r => r.isPrefixOf(l))),
    s"\n  L: ${ left.typ }\n  R: ${ right.typ }")

  def children: IndexedSeq[BaseIR] = Array(left, right)

  private val newRowType = left.typ.rowType.structInsert(right.typ.valueType, List(root))._1
  val typ: TableType = left.typ.copy(rowType = newRowType)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableLeftJoinRightDistinct(newLeft, newRight, root)
  }

  override def execute(hc: HailContext): TableValue = {
    val leftValue = left.execute(hc)
    val rightValue = right.execute(hc)

    leftValue.copy(
      typ = typ,
      rvd = leftValue.rvd.asInstanceOf[OrderedRVD]
        .orderedLeftJoinDistinctAndInsert(rightValue.rvd.asInstanceOf[OrderedRVD], root))
  }
}

// Must leave key fields unchanged. 'newKey' is used to rename key fields only.
case class TableMapRows(child: TableIR, newRow: IR, newKey: Option[IndexedSeq[String]]) extends TableIR {
  require(newKey.isDefined == child.typ.key.isDefined)
  require(newKey.forall(_.length == child.typ.key.get.length))
  val preservedKeyFields = child.typ.key.map(_.length)
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType = {
    val newRowType = newRow.typ.asInstanceOf[TStruct]
    child.typ.copy(rowType = newRowType, key = newKey)
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR], newKey)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val globalsBc = tv.globals.broadcast
    val gType = typ.globalType

    var scanInitNeedsGlobals = false
    var scanSeqNeedsGlobals = false
    var rowIterationNeedsGlobals = false

    val (scanAggs, scanInitOps, scanSeqOps, scanResultType, postScanIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", gType,
      "global", gType,
      "row", tv.typ.rowType,
      CompileWithAggregators.liftScan(newRow), "SCANR",
      { (nAggs: Int, initOp: IR) =>
        scanInitNeedsGlobals |= Mentions(initOp, "global")
        initOp
      },
      { (nAggs: Int, seqOp: IR) =>
        scanSeqNeedsGlobals |= Mentions(seqOp, "global")
        seqOp
      })

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long](
      "SCANR", scanResultType,
      "global", child.typ.globalType,
      "row", child.typ.rowType,
      postScanIR)
    assert(rTyp == typ.rowType)

    rowIterationNeedsGlobals |= Mentions(postScanIR, "global")

    val itF = if (scanAggs.nonEmpty) {
      Region.scoped { region =>
        val globals =
          if (scanInitNeedsGlobals) {
            val rvb = new RegionValueBuilder(region)
            rvb.start(gType)
            rvb.addAnnotation(gType, globalsBc.value)
            rvb.end()
          } else
            0
        scanInitOps(0)(region, scanAggs, globals, false)
      }

      val scanAggsPerPartition =
        tv.rvd.collectPerPartition { (i, ctx, it) =>
          val globals =
            if (scanSeqNeedsGlobals) {
              val rvb = new RegionValueBuilder(ctx.freshRegion)
              rvb.start(gType)
              rvb.addAnnotation(gType, globalsBc.value)
              rvb.end()
            } else
              0

          val scanSeqOpF = scanSeqOps(i)
          it.foreach { rv =>
            scanSeqOpF(rv.region, scanAggs, globals, false, rv.offset, false)
            ctx.region.clear()
          }
          scanAggs
        }.scanLeft(scanAggs) { (a1, a2) =>
          (a1, a2).zipped.map { (agg1, agg2) =>
            val newAgg = agg1.copy()
            newAgg.combOp(agg2)
            newAgg
          }
        }

      { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
        val partitionAggs = scanAggsPerPartition(i)

        val rvb = new RegionValueBuilder()
        val globals =
          if (rowIterationNeedsGlobals || scanSeqNeedsGlobals) {
            rvb.set(ctx.freshRegion)
            rvb.start(gType)
            rvb.addAnnotation(gType, globalsBc.value)
            rvb.end()
          } else
            0

        val rv2 = RegionValue()
        val newRow = f(i)
        val scanSeqOpF = scanSeqOps(i)
        it.map { rv =>
          rvb.set(rv.region)
          rvb.start(scanResultType)
          rvb.startStruct()
          var j = 0
          while (j < partitionAggs.length) {
            partitionAggs(j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val scanOffset = rvb.end()

          rv2.set(rv.region, newRow(rv.region, scanOffset, false, globals, false, rv.offset, false))
          scanSeqOpF(rv.region, partitionAggs, globals, false, rv.offset, false)
          rv2
        }
      }
    } else {
      { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
        val globals =
          if (rowIterationNeedsGlobals) {
            val rvb = new RegionValueBuilder(ctx.freshRegion)
            rvb.start(gType)
            rvb.addAnnotation(gType, globalsBc.value)
            rvb.end()
          } else
            0

        val rv2 = RegionValue()
        val newRow = f(i)
        it.map { rv =>
          rv2.set(rv.region, newRow(rv.region, 0, true, globals, false, rv.offset, false))
          rv2
        }
      }
    }

    val orvd = tv.rvd.toOrderedRVD
    val newRVD = orvd
      .truncateKey(orvd.typ.key.take(typ.rvdType.key.length))
      .mapPartitionsWithIndexPreservesPartitioning(typ.rvdType, itF)
      .toOldStyleRVD

    TableValue(typ, tv.globals, newRVD)
  }
}

case class TableMapGlobals(child: TableIR, newRow: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType =
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)

    val newGlobals = Interpret[Row](
      newRow,
      Env.empty[(Any, Type)].bind(
        "global" -> (tv.globals.value, child.typ.globalType)),
      FastIndexedSeq(),
      None)

    tv.copy(typ = typ, globals = BroadcastRow(newGlobals, typ.globalType, hc.sc))
  }
}


case class TableExplode(child: TableIR, fieldName: String) extends TableIR {
  assert(!child.typ.keyOrEmpty.contains(fieldName))

  def children: IndexedSeq[BaseIR] = Array(child)

  private val fieldIdx = child.typ.rowType.fieldIdx(fieldName)
  private val fieldType = child.typ.rowType.types(fieldIdx)
  private val rowType = child.typ.rowType.updateKey(fieldName, fieldIdx, fieldType.asInstanceOf[TContainer].elementType)

  val typ: TableType = child.typ.copy(rowType = rowType)

  def copy(newChildren: IndexedSeq[BaseIR]): TableExplode = {
    assert(newChildren.length == 1)
    TableExplode(newChildren(0).asInstanceOf[TableIR], fieldName)
  }

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val childRowType = child.typ.rowType

    val field = fieldType match {
      case TArray(_, _) =>
        GetField(Ref("row", childRowType), fieldName)
      case TSet(_, _) =>
        ToArray(GetField(Ref("row", childRowType), fieldName))
      case _ =>
        fatal(s"expected field to explode to be an array or set, found ${ fieldType }")
    }

    val (_, lengthF) = ir.Compile[Long, Int]("row", childRowType,
      ir.If(IsNA(field), ir.I32(0), ir.ArrayLen(field)))

    val (resultType, explodeF) = ir.Compile[Long, Int, Long]("row", childRowType,
      "i", TInt32(),
      ir.InsertFields(Ref("row", childRowType),
        Array(fieldName -> ir.ArrayRef(
          field,
          ir.Ref("i", TInt32())))))
    assert(resultType == typ.rowType)

    val itF: (Int, RVDContext, Iterator[RegionValue]) => Iterator[RegionValue] = { (i, ctx, it) =>
      val rv2 = RegionValue()
      val lenF = lengthF(i)
      val rowF = explodeF(i)
      it.flatMap { rv =>
        val n = lenF(rv.region, rv.offset, false)
        Iterator.range(0, n)
          .map { i =>
            val off = rowF(ctx.region, rv.offset, false, i, false)
            rv2.set(ctx.region, off)
            rv2
        }
      }
    }

    val orvd = prev.rvd.toOrderedRVD.boundary
    val adjKey = orvd.truncateKey(orvd.typ.key.takeWhile(_ != fieldName))
    val newRVD = adjKey.mapPartitionsWithIndexPreservesPartitioning(
      adjKey.typ.copy(rowType = rowType),
      itF)

    TableValue(typ, prev.globals, newRVD.toOldStyleRVD)
  }
}

case class TableUnion(children: IndexedSeq[TableIR]) extends TableIR {
  assert(children.nonEmpty)
  assert(children.tail.forall(_.typ.rowType == children(0).typ.rowType))
  assert(children.tail.forall(_.typ.key == children(0).typ.key))

  def copy(newChildren: IndexedSeq[BaseIR]): TableUnion = {
    TableUnion(newChildren.map(_.asInstanceOf[TableIR]))
  }

  val typ: TableType = children(0).typ

  def execute(hc: HailContext): TableValue = {
    val tvs = children.map(_.execute(hc))
    tvs(0).copy(
      rvd = if (typ.key.isDefined)
        OrderedRVD.union(tvs.map(_.rvd.asInstanceOf[OrderedRVD]))
      else
        RVD.union(tvs.map(_.rvd)))
  }
}

case class MatrixRowsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsTable = {
    assert(newChildren.length == 1)
    MatrixRowsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.rowsTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val rtv = mv.rowsTableValue
    assert(rtv.typ == typ)
    rtv
  }
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.colsTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val ctv = mv.colsTableValue
    assert(ctv.typ == typ)
    ctv
  }
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.entriesTableType

  def execute(hc: HailContext): TableValue = {
    val mv = child.execute(hc)
    val etv = mv.entriesTableValue
    assert(etv.typ == typ)
    etv
  }
}

case class TableDistinct(child: TableIR) extends TableIR {
  require(child.typ.key.isDefined)

  def children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableDistinct = {
    val IndexedSeq(newChild) = newChildren
    TableDistinct(newChild.asInstanceOf[TableIR])
  }

  val typ: TableType = child.typ

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    prev.copy(rvd = prev.rvd.asInstanceOf[OrderedRVD].distinctByKey())
  }
}

case class TableKeyByAndAggregate(
  child: TableIR,
  expr: IR,
  newKey: IR,
  nPartitions: Option[Int] = None,
  bufferSize: Int = 50) extends TableIR {
  require(expr.typ.isInstanceOf[TStruct])
  require(newKey.typ.isInstanceOf[TStruct])
  require(bufferSize > 0)

  def children: IndexedSeq[BaseIR] = Array(child, expr, newKey)

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyByAndAggregate = {
    val IndexedSeq(newChild: TableIR, newExpr: IR, newNewKey: IR) = newChildren
    TableKeyByAndAggregate(newChild, newExpr, newNewKey, nPartitions, bufferSize)
  }

  private val keyType = newKey.typ.asInstanceOf[TStruct]
  val typ: TableType = TableType(rowType = keyType ++ coerce[TStruct](expr.typ),
    globalType = child.typ.globalType,
    key = if (keyType.fieldNames.isEmpty) None else Some(keyType.fieldNames)
  )

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val (rvAggs, makeInit, makeSeq, aggResultType, postAggIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", child.typ.globalType,
      "global", child.typ.globalType,
      "row", child.typ.rowType,
      expr, "AGGR",
      (nAggs, initializeIR) => initializeIR,
      (nAggs, sequenceIR) => sequenceIR)

    val (rTyp, makeAnnotate) = ir.Compile[Long, Long, Long](
      "AGGR", aggResultType,
      "global", child.typ.globalType,
      postAggIR)

    val init = makeInit(0)
    Region.scoped { r =>
      val globalsOffset = prev.globals.toRegion(r)
      init(r, rvAggs, globalsOffset, false)
    }

    val nAggs = rvAggs.length

    assert(coerce[TStruct](rTyp) == typ.valueType, s"$rTyp, ${ typ.valueType }")

    val rowType = prev.typ.rowType
    val globalsType = prev.typ.globalType
    val globalsBc = prev.globals.broadcast

    val localKeyType = keyType
    val newValueType = typ.valueType
    val newRowType = typ.rowType
    val (_, makeKeyF) = ir.Compile[Long, Long, Long](
      "row", child.typ.rowType,
      "global", child.typ.globalType,
      newKey
    )

    val localBufferSize = bufferSize
    val combOp = { (aggs1: Array[RegionValueAggregator], aggs2: Array[RegionValueAggregator]) =>
      var i = 0
      while (i < aggs2.length) {
        aggs1(i).combOp(aggs2(i))
        i += 1
      }
      aggs1
    }

    val rdd = prev.rvd
      .boundary
      .mapPartitionsWithIndex { (i, ctx, it) =>
        val rvb = new RegionValueBuilder()
        val partRegion = ctx.freshContext.region

        rvb.set(partRegion)
        rvb.start(globalsType)
        rvb.addAnnotation(globalsType, globalsBc.value)
        val globals = rvb.end()

        val makeKey = {
          val f = makeKeyF(i)
          rv: RegionValue => {
            val keyOff = f(rv.region, rv.offset, false, globals, false)
            SafeRow.read(localKeyType, rv.region, keyOff).asInstanceOf[Row]
          }
        }
        val sequence = {
          val f = makeSeq(i)
          (rv: RegionValue, rvAggs: Array[RegionValueAggregator]) => {
            f(rv.region, rvAggs, globals, false, rv.offset, false)
          }
        }
        new BufferedAggregatorIterator[RegionValue, Array[RegionValueAggregator], Row](
          it,
          () => rvAggs.map(_.copy()),
          makeKey,
          sequence,
          localBufferSize)
      }.aggregateByKey(rvAggs, nPartitions.getOrElse(prev.rvd.getNumPartitions))(combOp, combOp)

    val crdd = ContextRDD.weaken(rdd).cmapPartitionsWithIndex(
      { (i, ctx, it) =>
        val region = ctx.region

        val rvb = new RegionValueBuilder()
        val partRegion = ctx.freshContext.region
        rvb.set(partRegion)
        rvb.start(globalsType)
        rvb.addAnnotation(globalsType, globalsBc.value)
        val globals = rvb.end()
        val annotate = makeAnnotate(i)

        val rv = RegionValue(region)
        it.map { case (key, aggs) =>

          rvb.set(region)
          rvb.start(aggResultType)
          rvb.startStruct()
          var j = 0
          while (j < nAggs) {
            aggs(j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val aggResultOff = rvb.end()

          rvb.start(newRowType)
          rvb.startStruct()
          var i = 0
          while (i < localKeyType.size) {
            rvb.addAnnotation(localKeyType.types(i), key.get(i))
            i += 1
          }

          val newValueOff = annotate(region,
            aggResultOff, false,
            globals, false)

          rvb.addAllFields(newValueType, region, newValueOff)

          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      })

    prev.copy(
      typ = typ,
      rvd = if (typ.key.isDefined)
        OrderedRVD.coerce(typ.rvdType, crdd)
      else
        UnpartitionedRVD(typ.rowType, crdd))
  }
}

// follows key_by non-empty key
case class TableAggregateByKey(child: TableIR, expr: IR) extends TableIR {
  require(child.typ.keyOrEmpty.nonEmpty)

  def children: IndexedSeq[BaseIR] = Array(child, expr)

  def copy(newChildren: IndexedSeq[BaseIR]): TableAggregateByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: TableIR, newExpr: IR) = newChildren
    TableAggregateByKey(newChild, newExpr)
  }

  val typ: TableType = child.typ.copy(rowType = child.typ.keyType.get ++ coerce[TStruct](expr.typ))

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    val prevRVD = prev.rvd.asInstanceOf[OrderedRVD]

    val (rvAggs, makeInit, makeSeq, aggResultType, postAggIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", child.typ.globalType,
      "global", child.typ.globalType,
      "row", child.typ.rowType,
      expr, "AGGR",
      (nAggs, initializeIR) => initializeIR,
      (nAggs, sequenceIR) => sequenceIR)

    val (rTyp, makeAnnotate) = ir.Compile[Long, Long, Long](
      "AGGR", aggResultType,
      "global", child.typ.globalType,
      postAggIR)

    val nAggs = rvAggs.length

    assert(coerce[TStruct](rTyp) == typ.valueType, s"$rTyp, ${ typ.valueType }")

    val rowType = prev.typ.rowType
    val keyType = prev.typ.keyType.get
    val keyIndices = prev.typ.keyFieldIdx.get
    val keyOrd = prevRVD.typ.kRowOrd
    val globalsType = prev.typ.globalType
    val globalsBc = prev.globals.broadcast

    val newValueType = typ.valueType
    val newRowType = typ.rowType
    val newOrvdType = prevRVD.typ.copy(rowType = newRowType)

    val newRVD = prevRVD.boundary.mapPartitionsWithIndexPreservesPartitioning(newOrvdType, { (i, ctx, it) =>
      val rvb = new RegionValueBuilder()
      val partRegion = ctx.freshContext.region

      rvb.set(partRegion)
      rvb.start(globalsType)
      rvb.addAnnotation(globalsType, globalsBc.value)
      val partGlobalsOff = rvb.end()

      val initialize = makeInit(i)
      val sequence = makeSeq(i)
      val annotate = makeAnnotate(i)

      new Iterator[RegionValue] {
        var isEnd = false
        var current: RegionValue = _
        val rowKey: WritableRegionValue = WritableRegionValue(keyType, ctx.freshRegion)
        val consumerRegion: Region = ctx.region
        val newRV = RegionValue(consumerRegion)

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

          rowKey.setSelect(rowType, keyIndices, current)

          rvAggs.foreach(_.clear())

          val region = current.region

          initialize(region, rvAggs, partGlobalsOff, false)

          do {
            val region = current.region

            sequence(region, rvAggs,
              partGlobalsOff, false,
              current.offset, false)
            current = null
          } while (hasNext && keyOrd.equiv(rowKey.value, current))

          rvb.set(consumerRegion)

          rvb.start(aggResultType)
          rvb.startStruct()
          var j = 0
          while (j < nAggs) {
            rvAggs(j).result(rvb)
            j += 1
          }
          rvb.endStruct()
          val aggResultOff = rvb.end()

          rvb.start(newRowType)
          rvb.startStruct()
          var i = 0
          while (i < keyType.size) {
            rvb.addField(keyType, rowKey.value, i)
            i += 1
          }

          val newValueOff = annotate(consumerRegion,
            aggResultOff, false,
            partGlobalsOff, false)

          rvb.addAllFields(newValueType, consumerRegion, newValueOff)

          rvb.endStruct()
          newRV.setOffset(rvb.end())
          newRV
        }
      }
    })

    prev.copy(rvd = newRVD, typ = typ)
  }
}

case class TableOrderBy(child: TableIR, sortFields: IndexedSeq[SortField]) extends TableIR {
  // TableOrderBy expects an unkeyed child, so that we can better optimize by
  // pushing these two steps around as needed
  require(child.typ.key.isEmpty)

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val IndexedSeq(newChild) = newChildren
    TableOrderBy(newChild.asInstanceOf[TableIR], sortFields)
  }

  val typ: TableType = child.typ.copy(key = None)

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val rowType = child.typ.rowType
    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.ordering
      (i, if (so == Ascending) fo else fo.reverse)
    }

    val ord: Ordering[Annotation] = new Ordering[Annotation] {
      def compare(a: Annotation, b: Annotation): Int = {
        var i = 0
        while (i < sortColIndexOrd.length) {
          val (fi, ford) = sortColIndexOrd(i)
          val c = ford.compare(
            a.asInstanceOf[Row].get(fi),
            b.asInstanceOf[Row].get(fi))
          if (c != 0) return c
          i += 1
        }

        0
      }
    }

    val act = implicitly[ClassTag[Annotation]]

    // FIXME: uses SafeRow, very bad!!!
    val rdd = prev.rdd.sortBy(identity[Annotation], ascending = true)(ord, act)
    val rvd = ContextRDD.weaken[RVDContext](rdd)
      .cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowType))
    TableValue(typ, prev.globals, UnpartitionedRVD(rowType, rvd))
  }
}

case class LocalizeEntries(child: MatrixIR, entriesFieldName: String) extends TableIR {
  private val m = Map(MatrixType.entriesIdentifier -> entriesFieldName)
  private val newRowType = child.typ.rvRowType.rename(m)

  def typ: TableType = TableType(newRowType, Some(child.typ.rowKey), child.typ.globalType)

  def children: IndexedSeq[BaseIR] = FastIndexedSeq(child)
  def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = {
    val IndexedSeq(newChild) = newChildren
    LocalizeEntries(newChild.asInstanceOf[MatrixIR], entriesFieldName)
  }

  def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    TableValue(typ, prev.globals, prev.rvd.updateType(prev.rvd.typ.copy(rowType = newRowType)))
  }
}
