package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr.ir.functions.{MatrixToTableFunction, TableToTableFunction}
import is.hail.expr.types._
import is.hail.expr.types.physical.{PInt32, PStruct}
import is.hail.expr.types.virtual._
import is.hail.expr.ir
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, BlockMatrixReadRowBlockedRDD}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.table.{AbstractTableSpec, Ascending, SortField}
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.{Formats, ShortTypeHints}

import scala.reflect.ClassTag

object TableIR {
  def read(hc: HailContext, path: String, dropRows: Boolean, requestedType: Option[TableType]): TableIR = {
    val successFile = path + "/_SUCCESS"
    if (!hc.hadoopConf.exists(path + "/_SUCCESS"))
      fatal(s"write failed: file not found: $successFile")

    val tr = TableNativeReader(path)
    TableRead(requestedType.getOrElse(tr.fullType), dropRows = dropRows, tr)
  }
}

abstract sealed class TableIR extends BaseIR {
  def typ: TableType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  protected[ir] def execute(hc: HailContext): TableValue =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR

  def persist(storageLevel: StorageLevel): TableIR = {
    val tv = Interpret(this)
    TableLiteral(tv.persist(storageLevel))
  }

  def unpersist(): TableIR = {
    val tv = Interpret(this)
    TableLiteral(tv.unpersist())
  }

  def pyPersist(storageLevel: String): TableIR = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel: $storageLevel")
    }
    persist(level)
  }

  def pyUnpersist(): TableIR = unpersist()

  def pyToDF(): DataFrame = Interpret(this).toDF()
}

case class TableLiteral(value: TableValue) extends TableIR {
  val typ: TableType = value.typ

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    TableLiteral(value)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = value
}

object TableReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[TableNativeReader],
      classOf[TextTableReader],
      classOf[TextInputFilterAndReplace],
      classOf[TableFromBlockMatrixNativeReader]))
}

abstract class TableReader {
  def apply(tr: TableRead): TableValue

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullType: TableType

  def fullRVDType: RVDType
}

case class TableNativeReader(path: String, var _spec: AbstractTableSpec = null) extends TableReader {
  lazy val spec = if (_spec != null)
    _spec
  else
    (RelationalSpec.read(HailContext.get, path): @unchecked) match {
      case ts: AbstractTableSpec => ts
      case _: AbstractMatrixTableSpec => fatal(s"file is a MatrixTable, not a Table: '$path'")
    }

  def partitionCounts: Option[IndexedSeq[Long]] = Some(spec.partitionCounts)

  def fullType: TableType = spec.table_type

  def fullRVDType: RVDType = fullType.canonicalRVDType

  def apply(tr: TableRead): TableValue = {
    val hc = HailContext.get

    val globals = spec.globalsComponent.readLocal(hc, path, tr.typ.globalType.physicalType)(0)
    val rvd = if (tr.dropRows)
      RVD.empty(hc.sc, tr.typ.canonicalRVDType)
    else {
      val rvd = spec.rowsComponent.read(hc, path, tr.typ.rowType.physicalType)
      if (rvd.typ.key startsWith tr.typ.key)
        rvd
      else {
        log.info("Sorting a table after read. Rewrite the table to prevent this in the future.")
        rvd.changeKey(tr.typ.key)
      }
    }
    TableValue(tr.typ, BroadcastRow(globals, tr.typ.globalType, hc.sc), rvd)
  }
}

case class TableFromBlockMatrixNativeReader(path: String, nPartitions: Option[Int] = None) extends TableReader {
  val metadata: BlockMatrixMetadata = BlockMatrix.readMetadata(HailContext.get, path)
  val getNumPartitions: Int = nPartitions.getOrElse(HailContext.get.sc.defaultMinPartitions)

  val partitionRanges = (0 until getNumPartitions).map { i =>
    val nRows = metadata.nRows
    val start = (i * nRows) / getNumPartitions
    val end = ((i + 1) * nRows) / getNumPartitions
    start until end
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = {
    Some(partitionRanges.map(r => r.end - r.start))
  }

  override def fullType: TableType = {
    val rowType = TStruct("row_idx" -> TInt64(), "entries" -> TArray(TFloat64()))
    TableType(rowType, Array("row_idx"), TStruct())
  }

  override def fullRVDType: RVDType = fullType.canonicalRVDType

  def apply(tr: TableRead): TableValue = {
    val rowsRDD = new BlockMatrixReadRowBlockedRDD(path, partitionRanges, metadata, HailContext.get)

    val partitionBounds = partitionRanges.map { r => Interval(Row(r.start), Row(r.end), true, false) }
    val partitioner = new RVDPartitioner(fullType.keyType, partitionBounds)

    val rvd = RVD(fullType.canonicalRVDType, partitioner, ContextRDD(rowsRDD))
    TableValue(fullType, BroadcastRow.empty(), rvd)
  }
}

case class TableRead(typ: TableType, dropRows: Boolean, tr: TableReader) extends TableIR {
  assert(PruneDeadFields.isSupertype(typ, tr.fullType),
    s"\n  original:  ${ tr.fullType }\n  requested: $typ")

  override def partitionCounts: Option[IndexedSeq[Long]] = tr.partitionCounts

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(typ, dropRows, tr)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = tr.apply(this)
}

case class TableParallelize(rowsAndGlobal: IR, nPartitions: Option[Int] = None) extends TableIR {
  require(rowsAndGlobal.typ.isInstanceOf[TStruct])
  require(rowsAndGlobal.typ.asInstanceOf[TStruct].fieldNames.sameElements(Array("rows", "global")))

  private val rowsType = rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("rows").asInstanceOf[TArray]
  private val globalsType = rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global").asInstanceOf[TStruct]

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(rowsAndGlobal)

  def copy(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    val IndexedSeq(newrowsAndGlobal: IR) = newChildren
    TableParallelize(newrowsAndGlobal, nPartitions)
  }

  val typ: TableType = TableType(
    rowsType.elementType.asInstanceOf[TStruct],
    FastIndexedSeq(),
    globalsType)

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val (Row(_rows: IndexedSeq[_], globals: Row), _) = CompileAndEvaluate[Row](rowsAndGlobal, optimize = false)
    val rows = _rows.asInstanceOf[IndexedSeq[Row]]
    rows.zipWithIndex.foreach { case (r, idx) =>
      if (r == null)
        fatal(s"cannot parallelize null values: found null value at index $idx")
    }

    log.info(s"parallelized ${ rows.length } rows")

    val rowTyp = typ.rowType.physicalType
    val rvd = ContextRDD.parallelize[RVDContext](hc.sc, rows, nPartitions)
      .cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowTyp))
    TableValue(typ, BroadcastRow(globals, typ.globalType, hc.sc), RVD.unkeyed(rowTyp, rvd))
  }
}

/**
  * Change the table to have key 'keys'.
  *
  * Let n be the longest common prefix of 'keys' and the old key, i.e. the
  * number of key fields that are not being changed.
  * - If 'isSorted', then 'child' must already be sorted by 'keys', and n must
  * not be zero. Thus, if 'isSorted', TableKeyBy will not shuffle or scan.
  * The new partitioner will be the old one with partition bounds truncated
  * to length n.
  * - If n = 'keys.length', i.e. we are simply shortening the key, do nothing
  * but change the table type to the new key. 'isSorted' is ignored.
  * - Otherwise, if 'isSorted' is false and n < 'keys.length', then shuffle.
  */
case class TableKeyBy(child: TableIR, keys: IndexedSeq[String], isSorted: Boolean = false) extends TableIR {
  private val fields = child.typ.rowType.fieldNames.toSet
  assert(keys.forall(fields.contains), s"${ keys.filter(k => !fields.contains(k)).mkString(", ") }")

  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = keys)

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, isSorted)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    tv.copy(typ = typ, rvd = tv.rvd.enforceKey(keys, isSorted))
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
    Array("idx"),
    TStruct.empty())

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val localRowType = typ.rowType
    val localPartCounts = partCounts
    val partStarts = partCounts.scanLeft(0)(_ + _)

    TableValue(typ,
      BroadcastRow(Row(), typ.globalType, hc.sc),
      new RVD(
        RVDType(typ.rowType.physicalType, Array("idx")),
        new RVDPartitioner(Array("idx"), typ.rowType,
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
                rvb.start(localRowType.physicalType)
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

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)

    if (pred == True())
      return tv
    else if (pred == False())
      return tv.copy(rvd = RVD.empty(hc.sc, typ.canonicalRVDType))

    val (rTyp, f) = ir.Compile[Long, Long, Boolean](
      "row", child.typ.rowType.physicalType,
      "global", child.typ.globalType.physicalType,
      pred)
    assert(rTyp.virtualType == TBoolean())

    tv.filterWithPartitionOp(f)((rowF, rv, globalRV) => rowF(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableHead(child: TableIR, n: Long) extends TableIR {
  require(n >= 0, fatal(s"TableHead: n must be non-negative! Found '$n'."))

  def typ: TableType = child.typ

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableHead = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableHead(newChild, n)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] =
    child.partitionCounts.map(getHeadPartitionCounts(_, n))

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    prev.copy(rvd = prev.rvd.head(n, child.partitionCounts))
  }
}

object RepartitionStrategy {
  val SHUFFLE: Int = 0
  val COALESCE: Int = 1
  val NAIVE_COALESCE: Int = 2
}

case class TableRepartition(child: TableIR, n: Int, strategy: Int) extends TableIR {
  def typ: TableType = child.typ

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableRepartition = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRepartition(newChild, n, strategy)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    val rvd = strategy match {
      case RepartitionStrategy.SHUFFLE => prev.rvd.coalesce(n, shuffle = true)
      case RepartitionStrategy.COALESCE => prev.rvd.coalesce(n, shuffle = false)
      case RepartitionStrategy.NAIVE_COALESCE => prev.rvd.naiveCoalesce(n)
    }

    prev.copy(rvd = rvd)
  }
}

object TableJoin {
  def apply(left: TableIR, right: TableIR, joinType: String): TableJoin =
    TableJoin(left, right, joinType, left.typ.key.length)
}

/**
  * Suppose 'left' has key [l_1, ..., l_n] and 'right' has key [r_1, ..., r_m].
  * Then [l_1, ..., l_j] and [r_1, ..., r_j] must have the same type, where
  * j = 'joinKey'. TableJoin computes the join of 'left' and 'right' along this
  * common prefix of their keys, returning a table with key
  * [l_1, ..., l_j, l_{j+1}, ..., l_n, r_{j+1}, ..., r_m].
  *
  * WARNING: If 'left' has any duplicate (full) key [k_1, ..., k_n], and j < m,
  * and 'right' has multiple rows with the corresponding join key
  * [k_1, ..., k_j] but distinct full keys, then the resulting table will have
  * out-of-order keys. To avoid this, ensure one of the following:
  * * j == m
  * * 'left' has distinct keys
  * * 'right' has distinct join keys (length j prefix), or at least no
  * distinct keys with the same join key.
  */
case class TableJoin(left: TableIR, right: TableIR, joinType: String, joinKey: Int)
  extends TableIR {

  require(joinKey >= 0)
  require(left.typ.key.length >= joinKey)
  require(right.typ.key.length >= joinKey)
  require(left.typ.keyType.truncate(joinKey) isIsomorphicTo right.typ.keyType.truncate(joinKey))
  require(left.typ.globalType.fieldNames.toSet
    .intersect(right.typ.globalType.fieldNames.toSet)
    .isEmpty)
  require(joinType == "inner" ||
    joinType == "left" ||
    joinType == "right" ||
    joinType == "outer" ||
    joinType == "zip")

  val children: IndexedSeq[BaseIR] = Array(left, right)

  private val leftRVDType =
    RVDType(left.typ.rowType.physicalType, left.typ.key.take(joinKey))
  private val rightRVDType =
    RVDType(right.typ.rowType.physicalType, right.typ.key.take(joinKey))

  require(leftRVDType.rowType.fieldNames.toSet
    .intersect(rightRVDType.valueType.fieldNames.toSet)
    .isEmpty)

  private val newRowType = leftRVDType.kType ++ leftRVDType.valueType ++ rightRVDType.valueType
  private val newGlobalType = left.typ.globalType ++ right.typ.globalType

  private val newKey = left.typ.key ++ right.typ.key.drop(joinKey)

  val typ: TableType = TableType(newRowType.virtualType, newKey, newGlobalType)
  val rvdType: RVDType = typ.canonicalRVDType

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

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val leftTV = left.execute(hc)
    val rightTV = right.execute(hc)

    val newGlobals = BroadcastRow(
      Row.merge(leftTV.globals.value, rightTV.globals.value),
      newGlobalType,
      leftTV.rvd.sparkContext)

    val joinedRVD = if (joinType == "zip") {
      val leftRVD = leftTV.rvd
      val rightRVD = rightTV.rvd
      leftRVD.orderedZipJoin(
        rightRVD,
        joinKey,
        rvMerger,
        RVDType(newRowType, newKey))
    } else {
      val leftRVD = leftTV.rvd
      val rightRVD = rightTV.rvd
      leftRVD.orderedJoin(
        rightRVD,
        joinKey,
        joinType,
        rvMerger,
        RVDType(newRowType, newKey))
    }

    TableValue(typ, newGlobals, joinedRVD)
  }
}

case class TableIntervalJoin(
  left: TableIR,
  right: TableIR,
  root: String,
  product: Boolean
) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  val rightType: Type = if (product) TArray(right.typ.valueType) else right.typ.valueType
  val (newRowPType, ins) = left.typ.rowType.physicalType.unsafeStructInsert(rightType.physicalType, List(root))
  val typ: TableType = left.typ.copy(rowType = newRowPType.virtualType)

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR =
    TableIntervalJoin(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[TableIR], root, product)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val leftValue = left.execute(hc)
    val rightValue = right.execute(hc)

    val rightRVDType = rightValue.rvd.typ

    val localNewRowPType = newRowPType
    val localIns = ins

    val joinedType = RVDType(newRowPType, typ.key)
    val newRVD =
      if (product) {
        val joiner = (_: RVDContext, it: Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => {
          val rvb = new RegionValueBuilder()
          val rv2 = RegionValue()
          it.map { case Muple(rv, is) =>
            rvb.set(rv.region)
            rvb.start(localNewRowPType)
            localIns(
              rv.region,
              rv.offset,
              rvb,
              () => {
                rvb.startArray(is.size)
                is.foreach(i => rvb.selectRegionValue(rightRVDType.rowType, rightRVDType.valueFieldIdx, i))
                rvb.endArray()
              })
            rv2.set(rv.region, rvb.end())

            rv2
          }
        }

        leftValue.rvd.orderedLeftIntervalJoin(rightValue.rvd, joiner, joinedType)
      } else {
        val joiner = (_: RVDContext, it: Iterator[JoinedRegionValue]) => {
          val rvb = new RegionValueBuilder()
          val rv2 = RegionValue()
          it.map { case Muple(rv, i) =>
            rvb.set(rv.region)
            rvb.start(localNewRowPType)
            localIns(
              rv.region,
              rv.offset,
              rvb,
              () =>
                if (i == null)
                  rvb.setMissing()
                else
                  rvb.selectRegionValue(rightRVDType.rowType, rightRVDType.valueFieldIdx, i))
            rv2.set(rv.region, rvb.end())

            rv2
          }
        }

        leftValue.rvd.orderedLeftIntervalJoinDistinct(rightValue.rvd, joiner, joinedType)
      }

    TableValue(typ, leftValue.globals, newRVD)
  }
}

case class TableZipUnchecked(left: TableIR, right: TableIR) extends TableIR {
  require((left.typ.rowType.fieldNames ++ right.typ.rowType.fieldNames).areDistinct())
  require(right.typ.key.isEmpty)

  val typ: TableType = left.typ.copy(rowType = left.typ.rowType ++ right.typ.rowType)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableZipUnchecked(newLeft, newRight)
  }

  override def execute(hc: HailContext): TableValue = {
    val tv1 = left.execute(hc)
    val tv2 = right.execute(hc)

    val inserter = InsertFields(
      Ref("left", left.typ.rowType),
      right.typ.rowType.fieldNames.map(f => f -> GetField(Ref("right", right.typ.rowType), f)))

    val rvdType: RVDType = RVDType(inserter.pType, tv1.rvd.typ.key)

    val (t2, makeF) = ir.Compile[Long, Long, Long](
      "left", tv1.rvd.typ.rowType,
      "right", tv2.rvd.typ.rowType,
      inserter)

    assert(t2.virtualType == typ.rowType)
    assert(t2 == rvdType.rowType)

    val rvd = tv1.rvd.zipPartitionsWithIndex(rvdType, tv2.rvd) { (i, ctx, it1, it2) =>
      val f = makeF(i)
      val region = ctx.region
      val rv3 = RegionValue(region)
      new Iterator[RegionValue] {
        def hasNext: Boolean = {
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
    TableValue(typ, tv1.globals, rvd)
  }
}

case class TableMultiWayZipJoin(children: IndexedSeq[TableIR], fieldName: String, globalName: String) extends TableIR {
  require(children.length > 0, "there must be at least one table as an argument")

  private val first = children.head
  private val rest = children.tail

  require(rest.forall(e => e.typ.rowType == first.typ.rowType), "all rows must have the same type")
  require(rest.forall(e => e.typ.key == first.typ.key), "all keys must be the same")
  require(rest.forall(e => e.typ.globalType == first.typ.globalType),
    "all globals must have the same type")

  private val newGlobalType = TStruct(globalName -> TArray(first.typ.globalType))
  private val newValueType = TStruct(fieldName -> TArray(first.typ.valueType))
  private val newRowType = first.typ.keyType ++ newValueType

  lazy val typ: TableType = first.typ.copy(
    rowType = newRowType,
    globalType = newGlobalType
  )

  val rvdType: RVDType = typ.canonicalRVDType

  def copy(newChildren: IndexedSeq[BaseIR]): TableMultiWayZipJoin =
    TableMultiWayZipJoin(newChildren.asInstanceOf[IndexedSeq[TableIR]], fieldName, globalName)

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val childValues = children.map(_.execute(hc))
    assert(childValues.map(_.rvd.typ).toSet.size == 1) // same physical types


    val childRVDs = childValues.map(_.rvd)
    val repartitionedRVDs =
      if (childRVDs(0).partitioner.satisfiesAllowedOverlap(typ.key.length - 1) &&
        childRVDs.forall(rvd => rvd.partitioner == childRVDs(0).partitioner))
        childRVDs.map(_.truncateKey(typ.key.length))
      else {
        info("TableMultiWayZipJoin: repartitioning children")
        val childRanges = childRVDs.flatMap(_.partitioner.rangeBounds)
        val newPartitioner = RVDPartitioner.generate(childRVDs.head.typ.kType.virtualType, childRanges)
        childRVDs.map(_.repartition(newPartitioner))
      }
    val newPartitioner = repartitionedRVDs(0).partitioner

    val rvdType = repartitionedRVDs(0).typ
    val rowType = rvdType.rowType
    val keyIdx = rvdType.kFieldIdx
    val valIdx = rvdType.valueFieldIdx
    val localRVDType = rvdType
    val localNewRowType = newRowType.physicalType
    val localDataLength = children.length
    val rvMerger = { (ctx: RVDContext, it: Iterator[ArrayBuilder[(RegionValue, Int)]]) =>
      val rvb = new RegionValueBuilder()
      val newRegionValue = RegionValue()

      it.map { rvs =>
        val rv = rvs(0)._1
        rvb.set(ctx.region)
        rvb.start(localNewRowType)
        rvb.startStruct()
        rvb.addFields(rowType, rv, keyIdx) // Add the key
        rvb.startMissingArray(localDataLength) // add the values
        var i = 0
        while (i < rvs.length) {
          val (rv, j) = rvs(i)
          rvb.setArrayIndex(j)
          rvb.setPresent()
          rvb.startStruct()
          rvb.addFields(rowType, rv, valIdx)
          rvb.endStruct()
          i += 1
        }
        rvb.endArrayUnchecked()
        rvb.endStruct()

        newRegionValue.set(rvb.region, rvb.end())
        newRegionValue
      }
    }

    val rvd = RVD(
      typ = RVDType(localNewRowType, typ.key),
      partitioner = newPartitioner,
      crdd = ContextRDD.czipNPartitions(repartitionedRVDs.map(_.crdd.boundary)) { (ctx, its) =>
        val orvIters = its.map(it => OrderedRVIterator(localRVDType, it, ctx))
        rvMerger(ctx, OrderedRVIterator.multiZipJoin(orvIters))
      })

    val newGlobals = BroadcastRow(
      Row(childValues.map(_.globals.value)),
      newGlobalType,
      childValues.head.rvd.sparkContext)

    TableValue(typ, newGlobals, rvd)
  }
}

case class TableLeftJoinRightDistinct(left: TableIR, right: TableIR, root: String) extends TableIR {
  require(right.typ.keyType isPrefixOf left.typ.keyType,
    s"\n  L: ${ left.typ }\n  R: ${ right.typ }")

  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  private val newRowType = left.typ.rowType.structInsert(right.typ.valueType, List(root))._1
  val typ: TableType = left.typ.copy(rowType = newRowType)
  val rvdType: RVDType = typ.canonicalRVDType

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  def copy(newChildren: IndexedSeq[BaseIR]): TableLeftJoinRightDistinct = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableLeftJoinRightDistinct(newLeft, newRight, root)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val leftValue = left.execute(hc)
    val rightValue = right.execute(hc)

    val joinKey = math.min(left.typ.key.length, right.typ.key.length)
    leftValue.copy(
      typ = typ,
      rvd = leftValue.rvd
        .orderedLeftJoinDistinctAndInsert(rightValue.rvd.truncateKey(joinKey), root))
  }
}

// Must leave key fields unchanged.
case class TableMapRows(child: TableIR, newRow: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType = child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val gType = typ.globalType

    var scanInitNeedsGlobals = false
    var scanSeqNeedsGlobals = false
    var rowIterationNeedsGlobals = false

    val (scanAggs, scanInitOps, scanSeqOps, scanResultType, postScanIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", gType.physicalType,
      "global", gType.physicalType,
      "row", tv.typ.rowType.physicalType,
      CompileWithAggregators.liftScan(newRow), "SCANR", { (nAggs: Int, initOp: IR) =>
        scanInitNeedsGlobals |= Mentions(initOp, "global")
        initOp
      }, { (nAggs: Int, seqOp: IR) =>
        scanSeqNeedsGlobals |= Mentions(seqOp, "global")
        seqOp
      })

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "row", child.typ.rowType.physicalType,
      "SCANR", scanResultType,
      postScanIR)
    assert(rTyp.virtualType == typ.rowType)

    rowIterationNeedsGlobals |= Mentions(postScanIR, "global")

    val globalsBc =
      if (rowIterationNeedsGlobals || scanInitNeedsGlobals || scanSeqNeedsGlobals)
        tv.globals.broadcast
      else
        null

    if (scanAggs.nonEmpty) {
      Region.scoped { region =>
        val globals =
          if (scanInitNeedsGlobals) {
            val rvb = new RegionValueBuilder(region)
            rvb.start(gType.physicalType)
            rvb.addAnnotation(gType, globalsBc.value)
            rvb.end()
          } else
            0
        scanInitOps(0)(region, scanAggs, globals, false)
      }

      val itF = { (i: Int, ctx: RVDContext, partitionAggs: Array[RegionValueAggregator], it: Iterator[RegionValue]) =>
        val rvb = new RegionValueBuilder()
        val globals =
          if (rowIterationNeedsGlobals || scanSeqNeedsGlobals) {
            rvb.set(ctx.freshRegion)
            rvb.start(gType.physicalType)
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
          rvb.startTuple()
          var j = 0
          while (j < partitionAggs.length) {
            partitionAggs(j).result(rvb)
            j += 1
          }
          rvb.endTuple()
          val scanOffset = rvb.end()

          rv2.set(rv.region, newRow(rv.region, globals, false, rv.offset, false, scanOffset, false))
          scanSeqOpF(rv.region, partitionAggs, globals, false, rv.offset, false)
          rv2
        }
      }

      val scanAggsPerPartition =
        tv.rvd.collectPerPartition { (i, ctx, it) =>
          val globals =
            if (scanSeqNeedsGlobals) {
              val rvb = new RegionValueBuilder(ctx.freshRegion)
              rvb.start(gType.physicalType)
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

      tv.copy(
        typ = typ,
        rvd = tv.rvd.mapPartitionsWithIndexAndValue(RVDType(rTyp.asInstanceOf[PStruct], typ.key), scanAggsPerPartition, itF))
    } else {
      val itF = { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
        val globals =
          if (rowIterationNeedsGlobals) {
            val rvb = new RegionValueBuilder(ctx.freshRegion)
            rvb.start(gType.physicalType)
            rvb.addAnnotation(gType, globalsBc.value)
            rvb.end()
          } else
            0

        val rv2 = RegionValue()
        val newRow = f(i)
        it.map { rv =>
          rv2.set(rv.region, newRow(rv.region, globals, false, rv.offset, false, 0, false))
          rv2
        }
      }

      tv.copy(
        typ = typ,
        rvd = tv.rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key), itF))
    }
  }
}

case class TableMapGlobals(child: TableIR, newGlobals: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newGlobals)

  val typ: TableType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)

    val (newGlobalValue, _) = CompileAndEvaluate[Row](newGlobals,
      Env("global" -> (tv.globals.value -> tv.globals.t)),
      FastIndexedSeq(),
      optimize = false)
    tv.copy(typ = typ, globals = BroadcastRow(newGlobalValue, typ.globalType, hc.sc))
  }
}

case class TableExplode(child: TableIR, path: IndexedSeq[String]) extends TableIR {
  assert(path.nonEmpty)
  assert(!child.typ.key.contains(path.head))

  lazy val children: IndexedSeq[BaseIR] = Array(child)

  private val childRowType = child.typ.rowType

  private val length: IR = {
    val lenUID = genUID()
    Let(lenUID,
      ArrayLen(ToArray(
        path.foldLeft[IR](Ref("row", childRowType))((struct, field) =>
          GetField(struct, field)))),
      If(IsNA(Ref(lenUID, TInt32())), 0, Ref(lenUID, TInt32())))
  }

  val idx = Ref(genUID(), TInt32())
  val newRow: InsertFields = {
    val refs = path.init.scanLeft(Ref("row", childRowType))((struct, name) =>
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

  val typ: TableType = child.typ.copy(rowType = newRow.typ)

  def copy(newChildren: IndexedSeq[BaseIR]): TableExplode = {
    assert(newChildren.length == 1)
    TableExplode(newChildren(0).asInstanceOf[TableIR], path)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val (_, l) = Compile[Long, Int]("row", prev.rvd.rowPType, length)
    val (t, f) = Compile[Long, Int, Long](
      "row", prev.rvd.rowPType,
      idx.name, PInt32(),
      newRow)
    assert(t.virtualType == typ.rowType)

    val rvdType: RVDType = RVDType(
      newRow.pType,
      prev.rvd.typ.key.takeWhile(_ != path.head)
    )
    TableValue(typ,
      prev.globals,
      prev.rvd.boundary.mapPartitionsWithIndex(rvdType, { (i, ctx, it) =>
        val region2 = ctx.region
        val rv2 = RegionValue(region2)
        val lenF = l(i)
        val rowF = f(i)
        it.flatMap { rv =>
          val len = lenF(rv.region, rv.offset, false)
          new Iterator[RegionValue] {
            private[this] var i = 0

            def hasNext: Boolean = i < len

            def next(): RegionValue = {
              rv2.setOffset(rowF(rv2.region, rv.offset, false, i, false))
              i += 1
              rv2
            }
          }
        }
      }))
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

  val rvdType: RVDType = typ.canonicalRVDType

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tvs = children.map(_.execute(hc))
    tvs(0).copy(
      rvd = RVD.union(tvs.map(_.rvd), tvs(0).typ.key.length))
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
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.colsTableType
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.entriesTableType
}

case class TableDistinct(child: TableIR) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableDistinct = {
    val IndexedSeq(newChild) = newChildren
    TableDistinct(newChild.asInstanceOf[TableIR])
  }

  val typ: TableType = child.typ

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    prev.copy(rvd = prev.rvd.truncateKey(prev.typ.key).distinctByKey())
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

  lazy val children: IndexedSeq[BaseIR] = Array(child, expr, newKey)

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyByAndAggregate = {
    val IndexedSeq(newChild: TableIR, newExpr: IR, newNewKey: IR) = newChildren
    TableKeyByAndAggregate(newChild, newExpr, newNewKey, nPartitions, bufferSize)
  }

  private val keyType = newKey.typ.asInstanceOf[TStruct]
  val typ: TableType = TableType(rowType = keyType ++ coerce[TStruct](expr.typ),
    globalType = child.typ.globalType,
    key = keyType.fieldNames
  )

  val rvdType: RVDType = typ.canonicalRVDType

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val (rvAggs, makeInit, makeSeq, aggResultType, postAggIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "global", child.typ.globalType.physicalType,
      "row", child.typ.rowType.physicalType,
      expr, "AGGR",
      (nAggs, initializeIR) => initializeIR,
      (nAggs, sequenceIR) => sequenceIR)

    val (rTyp, makeAnnotate) = ir.Compile[Long, Long, Long](
      "AGGR", aggResultType,
      "global", child.typ.globalType.physicalType,
      postAggIR)

    val init = makeInit(0)
    Region.scoped { r =>
      val globalsOffset = prev.globals.toRegion(r)
      init(r, rvAggs, globalsOffset, false)
    }

    val nAggs = rvAggs.length

    assert(rTyp.virtualType == typ.valueType, s"$rTyp, ${ typ.valueType }")

    val globalsType = prev.typ.globalType
    val globalsBc = prev.globals.broadcast

    val localKeyType = keyType
    val localKeyPType = keyType.physicalType
    val newRowType = typ.rowType.physicalType
    val (_, makeKeyF) = ir.Compile[Long, Long, Long](
      "row", child.typ.rowType.physicalType,
      "global", child.typ.globalType.physicalType,
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
        val partRegion = ctx.freshRegion

        rvb.set(partRegion)
        rvb.start(globalsType.physicalType)
        rvb.addAnnotation(globalsType, globalsBc.value)
        val globals = rvb.end()

        val makeKey = {
          val f = makeKeyF(i)
          rv: RegionValue => {
            val keyOff = f(rv.region, rv.offset, false, globals, false)
            SafeRow.read(localKeyPType, rv.region, keyOff).asInstanceOf[Row]
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
        val partRegion = ctx.freshRegion
        rvb.set(partRegion)
        rvb.start(globalsType.physicalType)
        rvb.addAnnotation(globalsType, globalsBc.value)
        val globals = rvb.end()
        val annotate = makeAnnotate(i)

        val rv = RegionValue(region)
        it.map { case (key, aggs) =>

          rvb.set(region)
          rvb.start(aggResultType)
          rvb.startTuple()
          var j = 0
          while (j < nAggs) {
            aggs(j).result(rvb)
            j += 1
          }
          rvb.endTuple()
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

          rvb.addAllFields(rTyp.asInstanceOf[PStruct], region, newValueOff)

          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      })

    prev.copy(
      typ = typ,
      rvd = RVD.coerce(RVDType(newRowType, keyType.fieldNames), crdd))
  }
}

// follows key_by non-empty key
case class TableAggregateByKey(child: TableIR, expr: IR) extends TableIR {
  require(child.typ.key.nonEmpty)

  lazy val children: IndexedSeq[BaseIR] = Array(child, expr)

  def copy(newChildren: IndexedSeq[BaseIR]): TableAggregateByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: TableIR, newExpr: IR) = newChildren
    TableAggregateByKey(newChild, newExpr)
  }

  val typ: TableType = child.typ.copy(rowType = child.typ.keyType ++ coerce[TStruct](expr.typ))

  val rvdType: RVDType = typ.canonicalRVDType

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)
    val prevRVD = prev.rvd

    val (rvAggs, makeInit, makeSeq, aggResultType, postAggIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "global", child.typ.globalType.physicalType,
      "row", child.typ.rowType.physicalType,
      expr, "AGGR",
      (nAggs, initializeIR) => initializeIR,
      (nAggs, sequenceIR) => sequenceIR)

    val (rTyp, makeAnnotate) = ir.Compile[Long, Long, Long](
      "global", child.typ.globalType.physicalType,
      "AGGR", aggResultType,
      postAggIR)

    val nAggs = rvAggs.length

    assert(rTyp.virtualType == typ.valueType, s"$rTyp, ${ typ.valueType }")

    val rowType = prev.typ.rowType
    val keyType = prev.typ.keyType
    val keyIndices = prev.typ.keyFieldIdx
    val keyOrd = prevRVD.typ.kRowOrd
    val globalsType = prev.typ.globalType
    val globalsBc = prev.globals.broadcast

    val newValueType = typ.valueType
    val newRowType = typ.rowType
    val newRVDType = prevRVD.typ.copy(rowType = newRowType.physicalType)

    val newRVD = prevRVD
      .repartition(prevRVD.partitioner.strictify)
      .boundary
      .mapPartitionsWithIndex(newRVDType, { (i, ctx, it) =>
        val rvb = new RegionValueBuilder()
        val partRegion = ctx.freshRegion

        rvb.set(partRegion)
        rvb.start(globalsType.physicalType)
        rvb.addAnnotation(globalsType, globalsBc.value)
        val partGlobalsOff = rvb.end()

        val initialize = makeInit(i)
        val sequence = makeSeq(i)
        val annotate = makeAnnotate(i)

        new Iterator[RegionValue] {
          var isEnd = false
          var current: RegionValue = _
          val rowKey: WritableRegionValue = WritableRegionValue(keyType.physicalType, ctx.freshRegion)
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

            rowKey.setSelect(rowType.physicalType, keyIndices, current)

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
            rvb.startTuple()
            var j = 0
            while (j < nAggs) {
              rvAggs(j).result(rvb)
              j += 1
            }
            rvb.endTuple()
            val aggResultOff = rvb.end()

            rvb.start(newRowType.physicalType)
            rvb.startStruct()
            var i = 0
            while (i < keyType.size) {
              rvb.addField(keyType.physicalType, rowKey.value, i)
              i += 1
            }

            val newValueOff = annotate(consumerRegion,
	      partGlobalsOff, false,
              aggResultOff, false)

            rvb.addAllFields(newValueType.physicalType, consumerRegion, newValueOff)

            rvb.endStruct()
            newRV.setOffset(rvb.end())
            newRV
          }
        }
      })

    prev.copy(rvd = newRVD, typ = typ)
  }
}

object TableOrderBy {
  def isAlreadyOrdered(sortFields: IndexedSeq[SortField], prevKey: IndexedSeq[String]): Boolean = {
    sortFields.length <= prevKey.length &&
      sortFields.zip(prevKey).forall { case (sf, k) =>
        sf.sortOrder == Ascending && sf.field == k
      }
  }
}

case class TableOrderBy(child: TableIR, sortFields: IndexedSeq[SortField]) extends TableIR {
  // TableOrderBy expects an unkeyed child, so that we can better optimize by
  // pushing these two steps around as needed

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableOrderBy = {
    val IndexedSeq(newChild) = newChildren
    TableOrderBy(newChild.asInstanceOf[TableIR], sortFields)
  }

  val typ: TableType = child.typ.copy(key = FastIndexedSeq())

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    val physicalKey = prev.rvd.typ.key
    if (TableOrderBy.isAlreadyOrdered(sortFields, physicalKey))
      return prev

    val rowType = child.typ.rowType
    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.ordering
      if (so == Ascending) fo else fo.reverse
    }.toArray

    val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

    val act = implicitly[ClassTag[Annotation]]

    val codec = RVD.wireCodec
    val rdd = prev.rvd.keyedEncodedRDD(codec, sortFields.map(_.field)).sortBy(_._1)(ord, act)
    val orderedCRDD = codec.decodeRDD(rowType.physicalType, rdd.map(_._2))
    TableValue(typ, prev.globals, RVD.unkeyed(rowType.physicalType, orderedCRDD))
  }
}

/** Create a Table from a MatrixTable, storing the column values in a global
  * field 'colsFieldName', and storing the entry values in a row field
  * 'entriesFieldName'.
  */
case class CastMatrixToTable(
  child: MatrixIR,
  entriesFieldName: String,
  colsFieldName: String
) extends TableIR {

  lazy val typ: TableType = LowerMatrixIR.loweredType(child.typ, entriesFieldName, colsFieldName)

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): CastMatrixToTable = {
    val IndexedSeq(newChild) = newChildren
    CastMatrixToTable(newChild.asInstanceOf[MatrixIR], entriesFieldName, colsFieldName)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts
}

case class TableRename(child: TableIR, rowMap: Map[String, String], globalMap: Map[String, String]) extends TableIR {
  require(rowMap.keys.forall(child.typ.rowType.hasField))
  require(globalMap.keys.forall(child.typ.globalType.hasField))

  def rowF(old: String): String = rowMap.getOrElse(old, old)

  def globalF(old: String): String = globalMap.getOrElse(old, old)

  lazy val typ: TableType = child.typ.copy(
    rowType = child.typ.rowType.rename(rowMap),
    globalType = child.typ.globalType.rename(globalMap),
    key = child.typ.key.map(k => rowMap.getOrElse(k, k))
  )

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableRename = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRename(newChild, rowMap, globalMap)
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val prev = child.execute(hc)

    TableValue(typ, prev.globals.copy(t = typ.globalType), prev.rvd.cast(typ.rowType.physicalType))
  }
}

case class TableFilterIntervals(child: TableIR, intervals: IndexedSeq[Interval], keep: Boolean) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableFilterIntervals(newChild, intervals, keep)
  }

  override lazy val typ: TableType = child.typ

  protected[ir] override def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val partitioner = RVDPartitioner.union(
      tv.typ.keyType,
      intervals,
      tv.rvd.typ.key.length - 1)
    TableValue(tv.typ, tv.globals, tv.rvd.filterIntervals(partitioner, keep))
  }
}

case class MatrixToTableApply(child: MatrixIR, function: MatrixToTableFunction) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToTableApply(newChild, function)
  }
  override lazy val typ: TableType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None
}

case class TableToTableApply(child: TableIR, function: TableToTableFunction) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None

  protected[ir] override def execute(hc: HailContext): TableValue = {
    function.execute(child.execute(hc))
  }
}

case class BlockMatrixToTable(child: BlockMatrixIR) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixToTable(newChild)
  }

  override val typ: TableType = {
    val rvType = TStruct("i" -> TInt64Optional, "j" -> TInt64Optional, "entry" -> TFloat64Optional)
    TableType(rvType, Array[String](), TStruct.empty())
  }

  protected[ir] override def execute(hc: HailContext): TableValue = {
    child.execute(hc).entriesTable()
  }
}

case class RelationalLetTable(name: String, value: IR, body: TableIR) extends TableIR {
  def typ: TableType = body.typ

  def children: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newValue: IR, newBody: TableIR) = newChildren
    RelationalLetTable(name, newValue, newBody)
  }
}