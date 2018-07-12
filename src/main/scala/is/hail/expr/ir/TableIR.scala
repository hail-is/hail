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
  assert(PruneDeadFields.isSupertype(typ, spec.table_type))

  override def partitionCounts: Option[IndexedSeq[Long]] = Some(spec.partitionCounts)

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(path, spec, typ, dropRows)
  }

  def execute(hc: HailContext): TableValue = {
    val globals = spec.globalsComponent.readLocal(hc, path, typ.globalType)(0)
    TableValue(typ,
      BroadcastRow(globals, typ.globalType, hc.sc),
      if (dropRows)
        UnpartitionedRVD.empty(hc.sc, typ.rowType)
      else
        spec.rowsComponent.read(hc, path, typ.rowType))
  }
}

case class TableParallelize(typ: TableType, rows: IndexedSeq[Row], nPartitions: Option[Int] = None) extends TableIR {
  assert(typ.globalType.size == 0)
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    assert(newChildren.isEmpty)
    TableParallelize(typ, rows, nPartitions)
  }

  def execute(hc: HailContext): TableValue = {
    val rowTyp = typ.rowType
    val rvd = ContextRDD.parallelize[RVDContext](hc.sc, rows, nPartitions)
      .cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, rowTyp))
    TableValue(typ, BroadcastRow(Row(), typ.globalType, hc.sc), new UnpartitionedRVD(rowTyp, rvd))
  }
}

case class TableImport(paths: Array[String], typ: TableType, readerOpts: TableReaderOptions) extends TableIR {
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
                fatal(s"""${ e.getClass.getName }: could not convert "$field" to $typ in column "$name" """)
            }
            i += 1
          }

          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }.value
      }
    }

    TableValue(typ, BroadcastRow(Row.empty, typ.globalType, hc.sc), new UnpartitionedRVD(rowTyp, rvd))
  }
}

case class TableKeyBy(child: TableIR, keys: IndexedSeq[String], nPartitionKeys: Option[Int], sort: Boolean = true) extends TableIR {
  private val fields = child.typ.rowType.fieldNames.toSet
  assert(keys.forall(fields.contains), s"${ keys.filter(k => !fields.contains(k)).mkString(", ") }")
  assert(nPartitionKeys.forall(_ <= keys.length))

  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = Some(keys))

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, nPartitionKeys, sort)
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val rvd = if (sort) {
      def resort: OrderedRVD = {
        val orvdType = new OrderedRVDType(nPartitionKeys.map(keys.take).getOrElse(keys).toArray, keys.toArray, typ.rowType)
        OrderedRVD.coerce(orvdType, tv.rvd, None, None)
      }

      tv.rvd match {
        case ordered: OrderedRVD =>
          if (ordered.typ.key.startsWith(keys) &&
            nPartitionKeys.getOrElse(keys.length) == ordered.typ.partitionKey.length)
            ordered.copy(typ = ordered.typ.copy(key = keys.toArray))
          else resort
        case _: UnpartitionedRVD =>
          resort
      }
    } else {
      tv.rvd match {
        case ordered: OrderedRVD => ordered.toUnpartitionedRVD
        case unordered: UnpartitionedRVD => unordered
      }
    }
    tv.copy(typ = typ, rvd = rvd)
  }
}

case class TableUnkey(child: TableIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = None)

  def copy(newChildren: IndexedSeq[BaseIR]): TableUnkey = {
    assert(newChildren.length == 1)
    TableUnkey(newChildren(0).asInstanceOf[TableIR])
  }

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val rvd = tv.rvd match {
      case ordered: OrderedRVD => ordered.toUnpartitionedRVD
      case unordered: UnpartitionedRVD => unordered
    }
    tv.copy(typ = typ, rvd = rvd)
  }
}

case class TableRange(n: Int, nPartitions: Int) extends TableIR {
  private val nPartitionsAdj = math.min(n, nPartitions)
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
        new OrderedRVDType(Array("idx"), Array("idx"), typ.rowType),
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
    val ktv = child.execute(hc)

    if (pred == True())
      return ktv
    else if (pred == False())
      return ktv.copy(rvd = ktv.rvd match {
        case orvd: OrderedRVD => OrderedRVD.empty(hc.sc, orvd.typ)
        case urvd: UnpartitionedRVD => UnpartitionedRVD.empty(hc.sc, urvd.rowType)
      })

    val (rTyp, f) = ir.Compile[Long, Long, Boolean](
      "row", child.typ.rowType,
      "global", child.typ.globalType,
      pred)
    assert(rTyp == TBoolean())
    ktv.filter((rv, globalRV) => f()(rv.region, rv.offset, false, globalRV.offset, false))
  }
}

case class TableJoin(left: TableIR, right: TableIR, joinType: String) extends TableIR {
  require(left.typ.keyType.zip(right.typ.keyType).exists { case (leftKey, rightKey) =>
    leftKey isIsomorphicTo rightKey
  })

  val children: IndexedSeq[BaseIR] = Array(left, right)

  private val joinedFields = left.typ.keyType.get.fields ++
    left.typ.valueType.fields ++
    right.typ.valueType.fields
  private val preNames = joinedFields.map(_.name).toArray
  private val (finalColumnNames, remapped) = mangle(preNames)

  val rightFieldMapping: Map[String, String] = {
    val remapMap = remapped.toMap
    (right.typ.key.get.iterator.zip(left.typ.key.get.iterator) ++
      right.typ.valueType.fieldNames.iterator.map(f => f -> remapMap.getOrElse(f, f))).toMap
  }

  val newRowType = TStruct(joinedFields.zipWithIndex.map {
    case (fd, i) => (finalColumnNames(i), fd.typ)
  }: _*)

  val typ: TableType = left.typ.copy(rowType = newRowType)

  def copy(newChildren: IndexedSeq[BaseIR]): TableJoin = {
    assert(newChildren.length == 2)
    TableJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      joinType)
  }

  def execute(hc: HailContext): TableValue = {
    val leftTV = left.execute(hc)
    val rightTV = right.execute(hc)
    val leftRowType = left.typ.rowType
    val rightRowType = right.typ.rowType
    val leftKeyFieldIdx = left.typ.keyFieldIdx.get
    val rightKeyFieldIdx = right.typ.keyFieldIdx.get
    val leftValueFieldIdx = left.typ.valueFieldIdx
    val rightValueFieldIdx = right.typ.valueFieldIdx
    val localNewRowType = newRowType
    val rvMerger = { (ctx: RVDContext, it: Iterator[JoinedRegionValue]) =>
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
    val leftORVD = leftTV.rvd match {
      case ordered: OrderedRVD => ordered
      case unordered =>
        OrderedRVD.coerce(
          new OrderedRVDType(left.typ.key.get.toArray, left.typ.key.get.toArray, leftRowType),
          unordered)
    }
    val rightORVD = rightTV.rvd match {
      case ordered: OrderedRVD => ordered
      case unordered =>
        val ordType =
          new OrderedRVDType(right.typ.key.get.toArray, right.typ.key.get.toArray, rightRowType)
        if (joinType == "left" || joinType == "inner")
          unordered.constrainToOrderedPartitioner(ordType, leftORVD.partitioner)
        else
          OrderedRVD.coerce(ordType, unordered, leftORVD.partitioner)
    }
    val joinedRVD = leftORVD.orderedJoin(
      rightORVD,
      joinType,
      rvMerger,
      new OrderedRVDType(leftORVD.typ.partitionKey, leftORVD.typ.key, newRowType))

    TableValue(typ, leftTV.globals, joinedRVD)
  }
}

// Must not modify key ordering.
// newKey is key of resulting Table, if newKey=None then result is unkeyed.
// preservedKeyFields is length of initial sequence of key fields whose values are unchanged.
// Thus if number of partition keys of underlying OrderedRVD is <= preservedKeyFields,
// partition bounds will remain valid.
case class TableMapRows(child: TableIR, newRow: IR, newKey: Option[IndexedSeq[String]], preservedKeyFields: Option[Int]) extends TableIR {
  require(!(newKey.isDefined ^ preservedKeyFields.isDefined))
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType = {
    val newRowType = newRow.typ.asInstanceOf[TStruct]
    child.typ.copy(rowType = newRowType, key = newKey)
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR], newKey, preservedKeyFields)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val globalsBc = tv.globals.broadcast
    val gType = typ.globalType

    val (scanAggs, scanInitOps, scanSeqOps, scanResultType, postScanIR) = ir.CompileWithAggregators[Long, Long, Long](
      "global", gType,
      "global", gType,
      "row", tv.typ.rowType,
      CompileWithAggregators.liftScan(newRow), "SCANR",
      (nAggs: Int, initOp: IR) => initOp,
      (nAggs: Int, seqOp: IR) => seqOp)

    val scanAggsPerPartition = if (scanAggs.nonEmpty) {
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        rvb.start(gType)
        rvb.addAnnotation(gType, globalsBc.value)
        val globals = rvb.end()
        scanInitOps()(region, scanAggs, globals, false)
      }
      tv.rvd.collectPerPartition { (ctx, it) =>
        val rvb = new RegionValueBuilder(ctx.freshRegion)
        rvb.start(gType)
        rvb.addAnnotation(gType, globalsBc.value)
        val globals = rvb.end()
        it.foreach { rv =>
          scanSeqOps()(rv.region, scanAggs, globals, false, rv.offset, false)
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
    } else Array.fill(tv.rvd.getNumPartitions)(Array.empty[RegionValueAggregator])

    val (rTyp, f) = ir.Compile[Long, Long, Long, Long](
      "SCANR", scanResultType,
      "global", child.typ.globalType,
      "row", child.typ.rowType,
      postScanIR)
    assert(rTyp == typ.rowType)

    val itF = { (i: Int, ctx: RVDContext, it: Iterator[RegionValue]) =>
      val partitionAggs = scanAggsPerPartition(i)
      val rvb = new RegionValueBuilder(ctx.freshRegion)
      rvb.start(gType)
      rvb.addAnnotation(gType, globalsBc.value)
      val globals = rvb.end()
      val rv2 = RegionValue()
      val newRow = f()
      it.map { rv =>
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
        scanSeqOps()(rv.region, partitionAggs, globals, false, rv.offset, false)
        rv2
      }
    }
    val newRVD = tv.rvd match {
      case ordered: OrderedRVD =>
        typ.key match {
          case Some(key) =>
            val pkLength = ordered.typ.partitionKey.length
            if (pkLength <= preservedKeyFields.get) {
              val newType = ordered.typ.copy(
                partitionKey = key.take(pkLength).toArray,
                key = key.toArray,
                rowType = typ.rowType)
              ordered.mapPartitionsWithIndexPreservesPartitioning(newType, itF)
            } else {
              val newType = ordered.typ.copy(
                partitionKey = key.toArray,
                key = key.toArray,
                rowType = typ.rowType)
              OrderedRVD.coerce(newType, ordered.mapPartitionsWithIndex(typ.rowType, itF))
            }
          case None =>
            ordered.mapPartitionsWithIndex(typ.rowType, itF)
        }
      case unordered: UnpartitionedRVD =>
        unordered.mapPartitionsWithIndex(typ.rowType, itF)
    }

    TableValue(typ, tv.globals, newRVD)
  }
}

case class TableMapGlobals(child: TableIR, newRow: IR, value: BroadcastRow) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  val typ: TableType =
    child.typ.copy(globalType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR], value)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  def execute(hc: HailContext): TableValue = {
    val tv = child.execute(hc)
    val gType = typ.globalType

    val (rTyp, f) = ir.Compile[Long, Long, Long](
      "global", child.typ.globalType,
      "value", value.t,
      newRow)
    assert(rTyp == gType)

    val newGlobals = Region.scoped { globalRegion =>
      val globalOff = tv.globals.toRegion(globalRegion)
      val valueOff = value.toRegion(globalRegion)
      val newOff = f()(globalRegion, globalOff, false, valueOff, false)

      tv.globals.copy(
        value = SafeRow(rTyp.asInstanceOf[TStruct], globalRegion, newOff),
        t = rTyp.asInstanceOf[TStruct])
    }

    TableValue(typ, newGlobals, tv.rvd)
  }
}


case class TableExplode(child: TableIR, fieldName: String) extends TableIR {
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

    val (_, isMissingF) = ir.Compile[Long, Boolean]("row", childRowType,
      ir.IsNA(field))

    val (_, lengthF) = ir.Compile[Long, Int]("row", childRowType,
      ir.ArrayLen(field))

    val (resultType, explodeF) = ir.Compile[Long, Int, Long]("row", childRowType,
      "i", TInt32(),
      ir.InsertFields(Ref("row", childRowType),
        Array(fieldName -> ir.ArrayRef(
          field,
          ir.Ref("i", TInt32())))))
    assert(resultType == typ.rowType)

    val itF: (RVDContext, Iterator[RegionValue]) => Iterator[RegionValue] = { (ctx, it) =>
      val rv2 = RegionValue()
      it.flatMap { rv =>
        val isMissing = isMissingF()(rv.region, rv.offset, false)
        if (isMissing)
          Iterator.empty
        else {
          val n = lengthF()(rv.region, rv.offset, false)
          Iterator.range(0, n)
            .map { i =>
              val off = explodeF()(ctx.region, rv.offset, false, i, false)
              rv2.set(ctx.region, off)
              rv2
            }
        }
      }
    }

    val newRVD: RVD = prev.rvd.boundary match {
      case rvd: UnpartitionedRVD =>
        rvd.mapPartitions(typ.rowType, itF)
      case orvd: OrderedRVD =>
        if (orvd.typ.key.contains(fieldName))
          orvd.mapPartitions(typ.rowType, itF)
        else
          orvd.mapPartitionsPreservesPartitioning(
            orvd.typ.copy(rowType = rowType),
            itF)
    }

    TableValue(typ, prev.globals, newRVD)
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
      rvd = RVD.union(tvs.map(_.rvd)))
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
    prev.copy(rvd = prev.enforceOrderingRVD.asInstanceOf[OrderedRVD].distinctByKey())
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

  val typ: TableType = child.typ.copy(rowType = child.typ.keyType.get.merge(coerce[TStruct](expr.typ))._1)

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

    val newRVD = prevRVD.boundary.mapPartitionsPreservesPartitioning(newOrvdType, { (ctx, it) =>
      val rvb = new RegionValueBuilder()
      val partRegion = ctx.freshContext.region

      rvb.set(partRegion)
      rvb.start(globalsType)
      rvb.addAnnotation(globalsType, globalsBc.value)
      val partGlobalsOff = rvb.end()

      val initialize = makeInit()
      val sequence = makeSeq()
      val annotate = makeAnnotate()

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
          rvb.set(region)
          rvb.start(globalsType)
          rvb.addRegionValue(globalsType, partRegion, partGlobalsOff)
          val globals = rvb.end()

          initialize(region, rvAggs, globals, false)

          do {
            val region = current.region
            rvb.set(region)
            rvb.start(globalsType)
            rvb.addRegionValue(globalsType, partRegion, partGlobalsOff)
            val globals = rvb.end()

            sequence(region, rvAggs,
              globals, false,
              current.offset, false)
            current = null
          } while (hasNext && keyOrd.equiv(rowKey.value, current))

          rvb.set(consumerRegion)

          rvb.start(globalsType)
          rvb.addRegionValue(globalsType, partRegion, partGlobalsOff)
          val globalOff = rvb.end()

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
            globalOff, false)

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
    TableValue(typ, prev.globals, new UnpartitionedRVD(rowType, rvd))
  }
}
