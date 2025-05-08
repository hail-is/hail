package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.agg.Aggs
import is.hail.expr.ir.compile.{Compile, CompileWithAggregators}
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.{RVDToTableStage, TableStage, TableStageToRVD}
import is.hail.io.{exportTypes, BufferSpec, ByteArrayDecoder, ByteArrayEncoder, TypedCodecSpec}
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.{
  PArray, PCanonicalArray, PCanonicalStruct, PInt32Required, PStruct, PType,
}
import is.hail.types.physical.stypes.{
  BooleanSingleCodeType, Int32SingleCodeType, PTypeReferenceSingleCodeType, StreamSingleCodeType,
}
import is.hail.types.physical.stypes.interfaces.NoBoxLongIterator
import is.hail.types.tcoerce
import is.hail.types.virtual.{Field, MatrixType, TArray, TInt32, TStream, TStruct, TableType}
import is.hail.utils._

import scala.reflect.ClassTag

import java.io.{DataInputStream, DataOutputStream}

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.StructType
import org.apache.spark.storage.StorageLevel

object TableExecuteIntermediate {
  def apply(tv: TableValue): TableExecuteIntermediate = new TableValueIntermediate(tv)

  def apply(ts: TableStage): TableExecuteIntermediate = new TableStageIntermediate(ts)
}

sealed trait TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage

  def asTableValue(ctx: ExecuteContext): TableValue

  def partitioner: RVDPartitioner
}

case class TableValueIntermediate(tv: TableValue) extends TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage =
    RVDToTableStage(tv.rvd, tv.globals.toEncodedLiteral(ctx.theHailClassLoader))

  def asTableValue(ctx: ExecuteContext): TableValue = tv

  def partitioner: RVDPartitioner = tv.rvd.partitioner
}

case class TableStageIntermediate(ts: TableStage) extends TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage = ts

  def asTableValue(ctx: ExecuteContext): TableValue = {
    val (globals, rvd) = TableStageToRVD(ctx, ts)
    TableValue(ctx, TableType(ts.rowType, ts.key, ts.globalType), globals, rvd)
  }

  def partitioner: RVDPartitioner = ts.partitioner
}

object TableValue {
  def apply(ctx: ExecuteContext, rowType: PStruct, key: IndexedSeq[String], rdd: ContextRDD[Long])
    : TableValue = {
    assert(rowType.required)
    val tt = TableType(rowType.virtualType, key, TStruct.empty)
    TableValue(ctx, tt, BroadcastRow.empty(ctx), RVD.coerce(ctx, RVDType(rowType, key), rdd))
  }

  def apply(
    ctx: ExecuteContext,
    rowType: TStruct,
    key: IndexedSeq[String],
    rdd: RDD[Row],
    rowPType: Option[PStruct] = None,
  ): TableValue = {
    val canonicalRowType = rowPType.getOrElse(
      PCanonicalStruct.canonical(rowType).setRequired(true).asInstanceOf[PStruct]
    )
    assert(canonicalRowType.required)
    val tt = TableType(rowType, key, TStruct.empty)
    TableValue(
      ctx,
      tt,
      BroadcastRow.empty(ctx),
      RVD.coerce(
        ctx,
        RVDType(canonicalRowType, key),
        ContextRDD.weaken(rdd).toRegionValues(canonicalRowType),
      ),
    )
  }

  def multiWayZipJoin(childValues: Seq[TableValue], fieldName: String, globalName: String)
    : TableValue = {
    def newGlobalType = TStruct(globalName -> TArray(childValues.head.typ.globalType))
    def newValueType = TStruct(fieldName -> TArray(childValues.head.typ.valueType))
    def newRowType = childValues.head.typ.keyType ++ newValueType
    val typ = childValues.head.typ.copy(
      rowType = newRowType,
      globalType = newGlobalType,
    )
    val ctx = childValues.head.ctx
    val sm = ctx.stateManager

    val childRVDs = RVD.unify(ctx, childValues.map(_.rvd)).toFastSeq
    assert(childRVDs.forall(_.typ.key.startsWith(typ.key)))

    val repartitionedRVDs =
      if (
        childRVDs(0).partitioner.satisfiesAllowedOverlap(typ.key.length - 1) &&
        childRVDs.forall(rvd => rvd.partitioner == childRVDs(0).partitioner)
      )
        childRVDs.map(_.truncateKey(typ.key.length))
      else {
        info("TableMultiWayZipJoin: repartitioning children")
        val childRanges = childRVDs.flatMap(_.partitioner.coarsenedRangeBounds(typ.key.length))
        val newPartitioner = RVDPartitioner.generate(ctx.stateManager, typ.keyType, childRanges)
        childRVDs.map(_.repartition(ctx, newPartitioner))
      }
    val newPartitioner = repartitionedRVDs(0).partitioner

    val rvdType = repartitionedRVDs(0).typ
    val rowType = rvdType.rowType
    val keyIdx = rvdType.kFieldIdx
    val valIdx = rvdType.valueFieldIdx
    val localRVDType = rvdType
    val keyFields = rvdType.kType.fields.map(f => (f.name, f.typ))
    val valueFields = rvdType.valueType.fields.map(f => (f.name, f.typ))
    val localNewRowType = PCanonicalStruct(
      required = true,
      keyFields ++ Array((
        fieldName,
        PCanonicalArray(
          PCanonicalStruct(required = false, valueFields: _*),
          required = true,
        ),
      )): _*
    )
    val localDataLength = childValues.length
    val rvMerger = { (ctx: RVDContext, it: Iterator[BoxedArrayBuilder[(RegionValue, Int)]]) =>
      val rvb = new RegionValueBuilder(sm)
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
      crdd = ContextRDD.czipNPartitions(repartitionedRVDs.map(_.crdd.toCRDDRegionValue)) {
        (ctx, its) =>
          val orvIters = its.map(it => OrderedRVIterator(localRVDType, it, ctx, sm))
          rvMerger(ctx, OrderedRVIterator.multiZipJoin(sm, orvIters))
      }.toCRDDPtr,
    )

    val newGlobals = BroadcastRow(ctx, Row(childValues.map(_.globals.javaValue)), newGlobalType)

    TableValue(ctx, typ, newGlobals, rvd)
  }

  def parallelize(ctx: ExecuteContext, rowsAndGlobal: IR, nPartitions: Option[Int]): TableValue = {
    val (ptype: PStruct, res) =
      CompileAndEvaluate._apply(ctx, rowsAndGlobal, optimize = false) match {
        case Right((t, off)) => (t.fields(0).typ, t.loadField(off, 0))
      }

    val globalsT = ptype.types(1).setRequired(true).asInstanceOf[PStruct]
    if (ptype.isFieldMissing(res, 1))
      fatal("'parallelize': found missing global value")
    val globals = BroadcastRow(ctx, RegionValue(ctx.r, ptype.loadField(res, 1)), globalsT)

    val rowsT = ptype.types(0).asInstanceOf[PArray]
    val rowT = rowsT.elementType.asInstanceOf[PStruct].setRequired(true)
    val spec = TypedCodecSpec(ctx, rowT, BufferSpec.wireSpec)

    val makeEnc = spec.buildEncoder(ctx, rowT)
    val rowsAddr = ptype.loadField(res, 0)
    val nRows = rowsT.loadLength(rowsAddr)

    val nSplits = math.min(nPartitions.getOrElse(16), math.max(nRows, 1))
    val parts = partition(nRows, nSplits)

    val bae = new ByteArrayEncoder(ctx.theHailClassLoader, makeEnc)
    var idx = 0
    val encRows = Array.tabulate(nSplits) { splitIdx =>
      val n = parts(splitIdx)
      bae.reset()
      val stop = idx + n
      while (idx < stop) {
        if (rowsT.isElementMissing(rowsAddr, idx))
          fatal(s"cannot parallelize null values: found null value at index $idx")
        bae.writeRegionValue(ctx.r, rowsT.loadElement(rowsAddr, idx))
        idx += 1
      }
      (n, bae.result())
    }

    def rowType = rowsAndGlobal.typ.asInstanceOf[TStruct]
      .fieldType("rows").asInstanceOf[TArray]
      .elementType.asInstanceOf[TStruct]
    val (resultRowType: PStruct, makeDec) = spec.buildDecoder(ctx, rowType)
    assert(
      resultRowType.virtualType == rowType,
      s"typ mismatch:" +
        s"\n  res=${resultRowType.virtualType}\n  typ=$rowType",
    )

    log.info(s"parallelized $nRows rows in $nSplits partitions")

    val rvd = ContextRDD.parallelize(encRows, encRows.length)
      .cmapPartitions { (ctx, it) =>
        it.flatMap { case (nRowPartition, arr) =>
          val bais = new ByteArrayDecoder(theHailClassLoaderForSparkWorkers, makeDec)
          bais.set(arr)
          Iterator.range(0, nRowPartition)
            .map(_ => bais.readValue(ctx.region))
        }
      }

    def globalsType =
      rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global").asInstanceOf[TStruct]
    val typ = TableType(rowType, FastSeq(), globalsType)

    TableValue(ctx, typ, globals, RVD.unkeyed(resultRowType, rvd))
  }

  def range(ctx: ExecuteContext, partCounts: IndexedSeq[Int]): TableValue = {
    val partStarts = partCounts.scanLeft(0)(_ + _)
    val nPartitions = partCounts.length
    val typ: TableType = TableType(
      TStruct("idx" -> TInt32),
      Array("idx"),
      TStruct.empty,
    )
    val rowType = PCanonicalStruct(true, "idx" -> PInt32Required)
    TableValue(
      ctx,
      typ,
      BroadcastRow.empty(ctx),
      new RVD(
        RVDType(rowType, Array("idx")),
        new RVDPartitioner(
          ctx.stateManager,
          Array("idx"),
          typ.rowType,
          Array.tabulate(nPartitions) { i =>
            val start = partStarts(i)
            val end = partStarts(i + 1)
            Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
          },
        ),
        ContextRDD.parallelize(Range(0, nPartitions), nPartitions)
          .cmapPartitionsWithIndex { case (i, ctx, _) =>
            val region = ctx.region

            val start = partStarts(i)
            Iterator.range(start, start + partCounts(i))
              .map { j =>
                val off = rowType.allocate(region)
                rowType.setFieldPresent(off, 0)
                Region.storeInt(rowType.fieldOffset(off, 0), j)
                off
              }
          },
      ),
    )
  }
}

case class TableValue(ctx: ExecuteContext, typ: TableType, globals: BroadcastRow, rvd: RVD) {
  if (typ.rowType != rvd.rowType)
    throw new RuntimeException(
      s"row mismatch:\n  typ: ${typ.rowType.parsableString()}\n  rvd: ${rvd.rowType.parsableString()}"
    )

  if (!rvd.typ.key.startsWith(typ.key))
    throw new RuntimeException(s"key mismatch:\n  typ: ${typ.key}\n  rvd: ${rvd.typ.key}")

  if (typ.globalType != globals.t.virtualType)
    throw new RuntimeException(
      s"globals mismatch:\n  typ: ${typ.globalType.parsableString()}\n  val: ${globals.t.virtualType.parsableString()}"
    )

  if (!globals.t.required)
    throw new RuntimeException(s"globals not required; ${globals.t}")

  def rdd: RDD[Row] =
    rvd.toRows

  def persist(ctx: ExecuteContext, level: StorageLevel) =
    TableValue(ctx, typ, globals, rvd.persist(ctx, level))

  def export(
    ctx: ExecuteContext,
    path: String,
    typesFile: String = null,
    header: Boolean = true,
    exportType: String = ExportType.CONCATENATED,
    delimiter: String = "\t",
  ): Unit = {
    val fs = ctx.fs
    fs.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, fs, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = rvd.rowPType
    val localTypes = fields.map(_.typ)

    val localDelim = delimiter
    rvd.mapPartitions { (ctx, it) =>
      val sb = new StringBuilder()

      it.map { ptr =>
        val ur = new UnsafeRow(localSignature, ctx.r, ptr)
        sb.clear()
        localTypes.indices.foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(ur.get(i), localTypes(i)))
        }(sb.append(localDelim))

        sb.result()
      }
    }.writeTable(
      ctx,
      path,
      Some(fields.map(_.name).mkString(localDelim)).filter(_ => header),
      exportType = exportType,
    )
  }

  def toDF(): DataFrame =
    HailContext.sparkBackend("toDF").sparkSession.createDataFrame(
      rvd.toRows,
      typ.rowType.schema.asInstanceOf[StructType],
    )

  def rename(globalMap: Map[String, String], rowMap: Map[String, String]): TableValue = {
    TableValue(
      ctx,
      typ.copy(
        rowType = typ.rowType.rename(rowMap),
        globalType = typ.globalType.rename(globalMap),
        key = typ.key.map(k => rowMap.getOrElse(k, k)),
      ),
      globals.copy(t = globals.t.rename(globalMap)),
      rvd = rvd.cast(rvd.rowPType.rename(rowMap)),
    )
  }

  def toMatrixValue(
    colKey: IndexedSeq[String],
    colsFieldName: String = LowerMatrixIR.colsFieldName,
    entriesFieldName: String = LowerMatrixIR.entriesFieldName,
  ): MatrixValue = {

    val (colType, colsFieldIdx) = typ.globalType.field(colsFieldName) match {
      case Field(_, TArray(t @ TStruct(_)), idx) => (t, idx)
      case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
    }

    val mType: MatrixType = MatrixType(
      typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
      colKey,
      colType,
      typ.key,
      typ.rowType.deleteKey(entriesFieldName),
      typ.rowType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[
        TArray
      ].elementType.asInstanceOf[TStruct],
    )

    val globalsT = globals.t
    val colsT = globalsT.field(colsFieldName).typ.asInstanceOf[PArray]

    val globals2 =
      if (colsT.required && colsT.elementType.required)
        globals
      else
        globals.cast(
          globalsT.insertFields(FastSeq(
            colsFieldName -> PCanonicalArray(colsT.elementType.setRequired(true), true)
          ))
        )

    val newTV = TableValue(ctx, typ, globals2, rvd)

    MatrixValue(
      mType,
      newTV.rename(
        Map(colsFieldName -> LowerMatrixIR.colsFieldName),
        Map(entriesFieldName -> LowerMatrixIR.entriesFieldName),
      ),
    )
  }

  def aggregateByKey(extracted: Aggs): TableValue = {
    val prevRVD = rvd.truncateKey(typ.key)
    val fsBc = ctx.fsBc
    val sm = ctx.stateManager

    val (_, makeInit) = CompileWithAggregators[AsmFunction2RegionLongUnit](
      ctx,
      extracted.states,
      FastSeq((
        TableIR.globalName,
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
      )),
      FastSeq(classInfo[Region], LongInfo),
      UnitInfo,
      extracted.init,
    )

    val (_, makeSeq) = CompileWithAggregators[AsmFunction3RegionLongLongUnit](
      ctx,
      extracted.states,
      FastSeq(
        (
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        ),
        (
          TableIR.rowName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prevRVD.rowPType)),
        ),
      ),
      FastSeq(classInfo[Region], LongInfo, LongInfo),
      UnitInfo,
      extracted.seqPerElt,
    )

    val valueIR = Let(FastSeq(extracted.resultRef.name -> extracted.results), extracted.postAggIR)
    val keyType = prevRVD.typ.kType

    val key = Ref(freshName(), keyType.virtualType)
    val value = Ref(freshName(), valueIR.typ)
    val (Some(PTypeReferenceSingleCodeType(rowType: PStruct)), makeRow) =
      CompileWithAggregators[AsmFunction3RegionLongLongLong](
        ctx,
        extracted.states,
        FastSeq(
          (
            TableIR.globalName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
          ),
          (key.name, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(keyType))),
        ),
        FastSeq(classInfo[Region], LongInfo, LongInfo),
        LongInfo,
        Let(
          FastSeq(value.name -> valueIR),
          InsertFields(
            key,
            tcoerce[TStruct](valueIR.typ).fieldNames.map(n => n -> GetField(value, n)),
          ),
        ),
      )

    val resultType = typ.copy(rowType = rowType.virtualType)

    val localChildRowType = prevRVD.rowPType
    val keyIndices = prevRVD.typ.kFieldIdx
    val keyOrd = prevRVD.typ.kRowOrd(ctx.stateManager)
    val globalsBc = globals.broadcast(ctx.theHailClassLoader)

    val newRVDType = prevRVD.typ.copy(rowType = rowType)

    val newRVD = prevRVD
      .repartition(ctx, prevRVD.partitioner.strictify())
      .boundary
      .mapPartitionsWithIndex(newRVDType) { (i, ctx, it) =>
        val partRegion = ctx.partitionRegion
        val globalsOff =
          globalsBc.value.readRegionValue(partRegion, theHailClassLoaderForSparkWorkers)

        val initialize = makeInit(
          theHailClassLoaderForSparkWorkers,
          fsBc.value,
          SparkTaskContext.get(),
          partRegion,
        )
        val sequence =
          makeSeq(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), partRegion)
        val newRowF =
          makeRow(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), partRegion)

        val aggRegion = ctx.freshRegion()

        new Iterator[Long] {
          var isEnd = false
          var current: Long = 0
          val rowKey: WritableRegionValue = WritableRegionValue(sm, keyType, ctx.freshRegion())
          val consumerRegion: Region = ctx.region

          def hasNext: Boolean = {
            if (isEnd || (current == 0 && !it.hasNext)) {
              isEnd = true
              return false
            }
            if (current == 0)
              current = it.next()
            true
          }

          def next(): Long = {
            if (!hasNext)
              throw new java.util.NoSuchElementException()

            rowKey.setSelect(localChildRowType, keyIndices, current, true)

            aggRegion.clear()
            initialize.newAggState(aggRegion)
            initialize(ctx.r, globalsOff)
            sequence.setAggState(aggRegion, initialize.getAggOffset())

            do {
              sequence(ctx.r, globalsOff, current)
              current = 0
            } while (hasNext && keyOrd.equiv(rowKey.value.offset, current))
            newRowF.setAggState(aggRegion, sequence.getAggOffset())

            newRowF(consumerRegion, globalsOff, rowKey.offset)
          }
        }
      }

    copy(rvd = newRVD, typ = resultType)
  }

  def explode(path: IndexedSeq[String]): TableValue = {
    val idx = Ref(freshName(), TInt32)

    val newRow: InsertFields = {
      val refs = path.init.scanLeft(Ref(TableIR.rowName, typ.rowType))((struct, name) =>
        Ref(freshName(), tcoerce[TStruct](struct.typ).field(name).typ)
      )

      path.zip(refs).zipWithIndex.foldRight[IR](idx) {
        case (((field, ref), i), arg) =>
          InsertFields(
            ref,
            FastSeq(field ->
              (if (i == refs.length - 1)
                 ArrayRef(CastToArray(GetField(ref, field)), arg)
               else
                 Let(FastSeq(refs(i + 1).name -> GetField(ref, field)), arg))),
          )
      }.asInstanceOf[InsertFields]
    }

    val length: IR =
      Coalesce(FastSeq(
        ArrayLen(CastToArray(
          path.foldLeft[IR](Ref(TableIR.rowName, typ.rowType))((struct, field) =>
            GetField(struct, field)
          )
        )),
        0,
      ))

    val (_, l) = Compile[AsmFunction2RegionLongInt](
      ctx,
      FastSeq((
        TableIR.rowName,
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
      )),
      FastSeq(classInfo[Region], LongInfo),
      IntInfo,
      length,
    )
    val (Some(PTypeReferenceSingleCodeType(newRowType: PStruct)), f) =
      Compile[AsmFunction3RegionLongIntLong](
        ctx,
        FastSeq(
          (
            TableIR.rowName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
          ),
          (idx.name, SingleCodeEmitParamType(true, Int32SingleCodeType)),
        ),
        FastSeq(classInfo[Region], LongInfo, IntInfo),
        LongInfo,
        newRow,
      )

    val rvdType: RVDType = RVDType(
      newRowType,
      rvd.typ.key.takeWhile(_ != path.head),
    )
    val fsBc = ctx.fsBc

    TableValue(
      ctx,
      typ.copy(rowType = newRow.typ),
      globals,
      rvd.boundary.mapPartitionsWithIndex(rvdType) { (i, ctx, it) =>
        val globalRegion = ctx.partitionRegion
        val lenF =
          l(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
        val rowF =
          f(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
        it.flatMap { ptr =>
          val len = lenF(ctx.region, ptr)
          new Iterator[Long] {
            private[this] var i = 0

            def hasNext: Boolean = i < len

            def next(): Long = {
              val ret = rowF(ctx.region, ptr, i)
              i += 1
              ret
            }
          }
        }
      },
    )
  }

  def filter(pred: IR): TableValue = {
    if (pred == True())
      return this
    else if (pred == False())
      return copy(rvd = RVD.empty(ctx, typ.canonicalRVDType))

    val (Some(BooleanSingleCodeType), f) = Compile[AsmFunction3RegionLongLongBoolean](
      ctx,
      FastSeq(
        (
          TableIR.rowName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
        ),
        (
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        ),
      ),
      FastSeq(classInfo[Region], LongInfo, LongInfo),
      BooleanInfo,
      Coalesce(FastSeq(pred, False())),
    )

    val fsBc = ctx.fsBc
    val localGlobals = globals.broadcast(ctx.theHailClassLoader)
    copy(rvd =
      rvd.filterWithContext[(AsmFunction3RegionLongLongBoolean, Long)](
        { (_, rvdCtx) =>
          val globalRegion = rvdCtx.partitionRegion
          (
            f(
              theHailClassLoaderForSparkWorkers,
              fsBc.value,
              SparkTaskContext.get(),
              globalRegion,
            ),
            localGlobals.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers),
          )
        },
        { case ((p, glob), ctx, ptr) => p(ctx.region, ptr, glob) },
      )
    )
  }

  def intervalJoin(rightValue: TableValue, root: String, product: Boolean): TableValue = {
    val leftRVDType = rvd.typ
    val rightRVDType = rightValue.rvd.typ.copy(key = rightValue.typ.key)
    val rightValueFields = rightRVDType.valueType.fieldNames

    val sm = ctx.stateManager
    val localKey = typ.key
    val localRoot = root
    val newRVD =
      if (product) {
        val joiner = (rightPType: PStruct) => {
          val leftRowType = leftRVDType.rowType
          val newRowType = leftRowType.appendKey(
            localRoot,
            PCanonicalArray(rightPType.selectFields(rightValueFields)),
          )
          (
            RVDType(newRowType, localKey),
            (_: RVDContext, it: Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => {
              val rvb = new RegionValueBuilder(sm)
              val rv2 = RegionValue()
              it.map { case Muple(rv, is) =>
                rvb.set(rv.region)
                rvb.start(newRowType)
                rvb.startStruct()
                rvb.addAllFields(leftRowType, rv)
                rvb.startArray(is.size)
                is.foreach(i => rvb.selectRegionValue(rightPType, rightRVDType.valueFieldIdx, i))
                rvb.endArray()
                rvb.endStruct()
                rv2.set(rv.region, rvb.end())

                rv2
              }
            },
          )
        }

        rvd.orderedLeftIntervalJoin(ctx, rightValue.rvd, joiner)
      } else {
        val joiner = (rightPType: PStruct) => {
          val leftRowType = leftRVDType.rowType
          val newRowType = leftRowType.appendKey(
            localRoot,
            rightPType.selectFields(rightValueFields).setRequired(false),
          )

          (
            RVDType(newRowType, localKey),
            (_: RVDContext, it: Iterator[JoinedRegionValue]) => {
              val rvb = new RegionValueBuilder(sm)
              val rv2 = RegionValue()
              it.map { case Muple(rv, i) =>
                rvb.set(rv.region)
                rvb.start(newRowType)
                rvb.startStruct()
                rvb.addAllFields(leftRowType, rv)
                if (i == null)
                  rvb.setMissing()
                else
                  rvb.selectRegionValue(rightPType, rightRVDType.valueFieldIdx, i)
                rvb.endStruct()
                rv2.set(rv.region, rvb.end())

                rv2
              }
            },
          )
        }

        rvd.orderedLeftIntervalJoinDistinct(ctx, rightValue.rvd, joiner)
      }

    val rightType = if (product) TArray(rightValue.typ.valueType) else rightValue.typ.valueType
    val newType = typ.copy(rowType = typ.rowType.appendKey(root, rightType))

    TableValue(ctx, newType, globals, newRVD)
  }

  def keyByAndAggregate(
    ctx: ExecuteContext,
    newKey: IR,
    extracted: Aggs,
    nPartitions: Option[Int],
    bufferSize: Int,
  ): TableValue = {
    val fsBc = ctx.fsBc
    val sm = ctx.stateManager

    val (Some(PTypeReferenceSingleCodeType(localKeyPType: PStruct)), makeKeyF) =
      Compile[AsmFunction3RegionLongLongLong](
        ctx,
        FastSeq(
          (
            TableIR.rowName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
          ),
          (
            TableIR.globalName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
          ),
        ),
        FastSeq(classInfo[Region], LongInfo, LongInfo),
        LongInfo,
        Coalesce(FastSeq(
          newKey,
          Die("Internal error: TableKeyByAndAggregate: newKey missing", newKey.typ),
        )),
      )

    val globalsBc = globals.broadcast(ctx.theHailClassLoader)

    val spec = BufferSpec.blockedUncompressed

    val (_, makeInit) = CompileWithAggregators[AsmFunction2RegionLongUnit](
      ctx,
      extracted.states,
      FastSeq((
        TableIR.globalName,
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
      )),
      FastSeq(classInfo[Region], LongInfo),
      UnitInfo,
      extracted.init,
    )

    val (_, makeSeq) = CompileWithAggregators[AsmFunction3RegionLongLongUnit](
      ctx,
      extracted.states,
      FastSeq(
        (
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        ),
        (
          TableIR.rowName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
        ),
      ),
      FastSeq(classInfo[Region], LongInfo, LongInfo),
      UnitInfo,
      extracted.seqPerElt,
    )

    val (Some(PTypeReferenceSingleCodeType(rTyp: PStruct)), makeAnnotate) =
      CompileWithAggregators[AsmFunction2RegionLongLong](
        ctx,
        extracted.states,
        FastSeq((
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        )),
        FastSeq(classInfo[Region], LongInfo),
        LongInfo,
        Let(FastSeq(extracted.resultRef.name -> extracted.results), extracted.postAggIR),
      )

    val serialize = extracted.serialize(ctx, spec)
    val deserialize = extracted.deserialize(ctx, spec)
    val combOp = extracted.combOpFSerializedWorkersOnly(ctx, spec)

    val hcl = theHailClassLoaderForSparkWorkers
    val tc = ctx.taskContext
    val initF = makeInit(hcl, fsBc.value, tc, ctx.r)
    val globalsOffset = globals.value.offset
    val initAggs = ctx.r.pool.scopedRegion { aggRegion =>
      initF.newAggState(aggRegion)
      initF(ctx.r, globalsOffset)
      serialize(hcl, tc, aggRegion, initF.getAggOffset())
    }

    val newRowType = PCanonicalStruct(
      required = true,
      localKeyPType.fields.map(f => (f.name, PType.canonical(f.typ))) ++ rTyp.fields.map(f =>
        (f.name, f.typ)
      ): _*
    )

    val localBufferSize = bufferSize
    val rdd = rvd
      .boundary
      .mapPartitionsWithIndex { (i, ctx, it) =>
        val partRegion = ctx.partitionRegion
        val hcl = theHailClassLoaderForSparkWorkers
        val tc = SparkTaskContext.get()
        val globals = globalsBc.value.readRegionValue(partRegion, hcl)
        val makeKey = {
          val f = makeKeyF(hcl, fsBc.value, tc, partRegion)
          ptr: Long => {
            val keyOff = f(ctx.region, ptr, globals)
            SafeRow.read(localKeyPType, keyOff).asInstanceOf[Row]
          }
        }
        val makeAgg = { () =>
          val aggRegion = ctx.freshRegion()
          RegionValue(aggRegion, deserialize(hcl, tc, aggRegion, initAggs))
        }

        val seqOp = {
          val f = makeSeq(hcl, fsBc.value, SparkTaskContext.get(), partRegion)
          (ptr: Long, agg: RegionValue) => {
            f.setAggState(agg.region, agg.offset)
            f(ctx.region, globals, ptr)
            agg.setOffset(f.getAggOffset())
            ctx.region.clear()
          }
        }
        val serializeAndCleanupAggs = { rv: RegionValue =>
          val a = serialize(hcl, tc, rv.region, rv.offset)
          rv.region.close()
          a
        }

        new BufferedAggregatorIterator[Long, RegionValue, Array[Byte], Row](
          it,
          makeAgg,
          makeKey,
          seqOp,
          serializeAndCleanupAggs,
          localBufferSize)
      }.aggregateByKey(initAggs, nPartitions.getOrElse(rvd.getNumPartitions))(combOp, combOp)

    val keyType = tcoerce[TStruct](newKey.typ)
    val crdd = ContextRDD.weaken(rdd).cmapPartitionsWithIndex({ (i, ctx, it) =>
      val region = ctx.region

      val rvb = new RegionValueBuilder(sm)
      val partRegion = ctx.partitionRegion
      val hcl = theHailClassLoaderForSparkWorkers
      val tc = SparkTaskContext.get()
      val globals = globalsBc.value.readRegionValue(partRegion, hcl)
      val annotate = makeAnnotate(hcl, fsBc.value, tc, partRegion)

      it.map { case (key, aggs) =>
        rvb.set(region)
        rvb.start(newRowType)
        rvb.startStruct()
        var i = 0
        while (i < keyType.size) {
          rvb.addAnnotation(keyType.types(i), key.get(i))
          i += 1
        }

        val aggOff = deserialize(hcl, tc, region, aggs)
        annotate.setAggState(region, aggOff)
        rvb.addAllFields(rTyp, region, annotate(region, globals))
        rvb.endStruct()
        rvb.end()
      }
    })

    val newType: TableType = typ.copy(
      rowType = newRowType.virtualType,
      key = keyType.fieldNames,
    )

    copy(
      typ = newType,
      rvd = RVD.coerce(ctx, RVDType(newRowType, keyType.fieldNames), crdd),
    )
  }

  def mapGlobals(newGlobals: IR): TableValue = {
    val (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) =
      Compile[AsmFunction2RegionLongLong](
        ctx,
        FastSeq((
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        )),
        FastSeq(classInfo[Region], LongInfo),
        LongInfo,
        Coalesce(FastSeq(
          newGlobals,
          Die("Internal error: TableMapGlobals: globals missing", newGlobals.typ),
        )),
      )

    val resultOff =
      f(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)(ctx.r, globals.value.offset)
    val newType = typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

    copy(typ = newType, globals = BroadcastRow(ctx, RegionValue(ctx.r, resultOff), resultPType))
  }

  def mapPartitions(globalName: Name, partitionStreamName: Name, body: IR, allowedOverlap: Int)
    : TableValue = {
    val rowPType = rvd.rowPType
    val globalPType = globals.t

    val (newRowPType: PStruct, makeIterator) = CompileIterator.forTableMapPartitions(
      ctx,
      globalPType,
      rowPType,
      Subst(
        body,
        BindingEnv(Env(
          globalName -> In(
            0,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globalPType)),
          ),
          partitionStreamName -> In(
            1,
            SingleCodeEmitParamType(
              true,
              StreamSingleCodeType(requiresMemoryManagementPerElement = true, rowPType, true),
            ),
          ),
        )),
      ),
    )

    val globalsBc = globals.broadcast(ctx.theHailClassLoader)

    val fsBc = ctx.fsBc
    val itF = { (idx: Int, consumerCtx: RVDContext, partition: (RVDContext) => Iterator[Long]) =>
      val boxedPartition = new NoBoxLongIterator {
        var eos: Boolean = false
        var iter: Iterator[Long] = _
        override def init(partitionRegion: Region, elementRegion: Region): Unit =
          iter = partition(new RVDContext(partitionRegion, elementRegion))

        override def next(): Long =
          if (!iter.hasNext) {
            eos = true
            0L
          } else
            iter.next()

        override def close(): Unit = ()
      }
      makeIterator(
        theHailClassLoaderForSparkWorkers,
        fsBc.value,
        SparkTaskContext.get(),
        consumerCtx,
        globalsBc.value.readRegionValue(
          consumerCtx.partitionRegion,
          theHailClassLoaderForSparkWorkers,
        ),
        boxedPartition,
      ).map(l => l.longValue())
    }

    val newRVD = rvd.repartition(ctx, rvd.partitioner.strictify(allowedOverlap))
    val newType = typ.copy(
      rowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
    )

    copy(
      typ = newType,
      rvd = newRVD
        .mapPartitionsWithContextAndIndex(RVDType(newRowPType, typ.key))(itF),
    )
  }

  def mapRows(extracted: Aggs): TableValue = {
    val fsBc = ctx.fsBc
    val newType = typ.copy(rowType = extracted.postAggIR.typ.asInstanceOf[TStruct])

    if (extracted.aggs.isEmpty) {
      val (Some(PTypeReferenceSingleCodeType(rTyp)), f) =
        Compile[AsmFunction3RegionLongLongLong](
          ctx,
          FastSeq(
            (
              TableIR.globalName,
              SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
            ),
            (
              TableIR.rowName,
              SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
            ),
          ),
          FastSeq(classInfo[Region], LongInfo, LongInfo),
          LongInfo,
          Coalesce(FastSeq(
            extracted.postAggIR,
            Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ),
          )),
        )

      val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, TableIR.globalName)
      val globalsBc =
        if (rowIterationNeedsGlobals)
          globals.broadcast(ctx.theHailClassLoader)
        else
          null

      val fsBc = ctx.fsBc
      val itF = { (i: Int, ctx: RVDContext, it: Iterator[Long]) =>
        val globalRegion = ctx.partitionRegion
        val globals = if (rowIterationNeedsGlobals)
          globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
        else
          0

        val newRow =
          f(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
        it.map(ptr => newRow(ctx.r, globals, ptr))
      }

      copy(
        typ = newType,
        rvd = rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key))(itF),
      )
    }

    val scanInitNeedsGlobals = Mentions(extracted.init, TableIR.globalName)
    val scanSeqNeedsGlobals = Mentions(extracted.seqPerElt, TableIR.globalName)
    val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, TableIR.globalName)

    val globalsBc =
      if (rowIterationNeedsGlobals || scanInitNeedsGlobals || scanSeqNeedsGlobals)
        globals.broadcast(ctx.theHailClassLoader)
      else
        null

    val spec = BufferSpec.blockedUncompressed

    // Order of operations:
    // 1. init op on all aggs and serialize to byte array.
    // 2. load in init op on each partition, seq op over partition, serialize.
    // 3. load in partition aggregations, comb op as necessary, serialize.
    // 4. load in partStarts, calculate newRow based on those results.

    val (_, initF) = CompileWithAggregators[AsmFunction2RegionLongUnit](
      ctx,
      extracted.states,
      FastSeq((
        TableIR.globalName,
        SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
      )),
      FastSeq(classInfo[Region], LongInfo),
      UnitInfo,
      Begin(FastSeq(extracted.init)),
    )

    val serializeF = extracted.serialize(ctx, spec)

    val (_, eltSeqF) = CompileWithAggregators[AsmFunction3RegionLongLongUnit](
      ctx,
      extracted.states,
      FastSeq(
        (
          TableIR.globalName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
        ),
        (
          TableIR.rowName,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
        ),
      ),
      FastSeq(classInfo[Region], LongInfo, LongInfo),
      UnitInfo,
      extracted.seqPerElt,
    )

    val read = extracted.deserialize(ctx, spec)
    val write = extracted.serialize(ctx, spec)
    val combOpFNeedsPool = extracted.combOpFSerializedFromRegionPool(ctx, spec)

    val (Some(PTypeReferenceSingleCodeType(rTyp)), f) =
      CompileWithAggregators[AsmFunction3RegionLongLongLong](
        ctx,
        extracted.states,
        FastSeq(
          (
            TableIR.globalName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globals.t)),
          ),
          (
            TableIR.rowName,
            SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
          ),
        ),
        FastSeq(classInfo[Region], LongInfo, LongInfo),
        LongInfo,
        Let(
          FastSeq(extracted.resultRef.name -> extracted.results),
          Coalesce(FastSeq(
            extracted.postAggIR,
            Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ),
          )),
        ),
      )

    // 1. init op on all aggs and write out to initPath
    val initAgg = ctx.r.pool.scopedRegion { aggRegion =>
      ctx.r.pool.scopedRegion { fRegion =>
        val init = initF(ctx.theHailClassLoader, fsBc.value, ctx.taskContext, fRegion)
        init.newAggState(aggRegion)
        init(fRegion, globals.value.offset)
        serializeF(ctx.theHailClassLoader, ctx.taskContext, aggRegion, init.getAggOffset())
      }
    }

    if (ctx.getFlag("distributed_scan_comb_op") != null && extracted.shouldTreeAggregate) {
      val fsBc = ctx.fs.broadcast
      val tmpBase = ctx.createTmpPath("table-map-rows-distributed-scan")
      val d = digitsNeeded(rvd.getNumPartitions)
      val files = rvd.mapPartitionsWithIndex { (i, ctx, it) =>
        val path = tmpBase + "/" + partFile(d, i, TaskContext.get)
        val globalRegion = ctx.freshRegion()
        val globals = if (scanSeqNeedsGlobals)
          globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
        else 0

        ctx.r.pool.scopedSmallRegion { aggRegion =>
          val tc = SparkTaskContext.get()
          val seq = eltSeqF(theHailClassLoaderForSparkWorkers, fsBc.value, tc, globalRegion)

          seq.setAggState(
            aggRegion,
            read(theHailClassLoaderForSparkWorkers, tc, aggRegion, initAgg),
          )
          it.foreach { ptr =>
            seq(ctx.region, globals, ptr)
            ctx.region.clear()
          }
          using(new DataOutputStream(fsBc.value.create(path))) { os =>
            val bytes = write(theHailClassLoaderForSparkWorkers, tc, aggRegion, seq.getAggOffset())
            os.writeInt(bytes.length)
            os.write(bytes)
          }
          Iterator.single(path)
        }
      }.collect()

      val fileStack = new BoxedArrayBuilder[Array[String]]()
      var filesToMerge: Array[String] = files
      while (filesToMerge.length > 1) {
        val nToMerge = filesToMerge.length / 2
        log.info(s"Running distributed combine stage with $nToMerge tasks")
        fileStack += filesToMerge

        filesToMerge =
          ContextRDD.weaken(SparkBackend.sparkContext("TableMapRows.execute").parallelize(
            0 until nToMerge,
            nToMerge,
          ))
            .cmapPartitions { (ctx, it) =>
              val i = it.next()
              assert(it.isEmpty)
              val path = tmpBase + "/" + partFile(d, i, TaskContext.get)
              val file1 = filesToMerge(i * 2)
              val file2 = filesToMerge(i * 2 + 1)

              def readToBytes(is: DataInputStream): Array[Byte] = {
                val len = is.readInt()
                val b = new Array[Byte](len)
                is.readFully(b)
                b
              }

              val b1 = using(new DataInputStream(fsBc.value.open(file1)))(readToBytes)
              val b2 = using(new DataInputStream(fsBc.value.open(file2)))(readToBytes)
              using(new DataOutputStream(fsBc.value.create(path))) { os =>
                val bytes = combOpFNeedsPool(() =>
                  (ctx.r.pool, theHailClassLoaderForSparkWorkers, SparkTaskContext.get())
                )(b1, b2)
                os.writeInt(bytes.length)
                os.write(bytes)
              }
              Iterator.single(path)
            }.collect()
      }
      fileStack += filesToMerge

      val itF = { (i: Int, ctx: RVDContext, it: Iterator[Long]) =>
        val globalRegion = ctx.freshRegion()
        val globals = if (rowIterationNeedsGlobals || scanSeqNeedsGlobals)
          globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
        else
          0
        val partitionAggs = {
          var j = 0
          var x = i
          val ab = new BoxedArrayBuilder[String]
          while (j < fileStack.length) {
            assert(x <= fileStack(j).length)
            if (x % 2 != 0) {
              x -= 1
              ab += fileStack(j)(x)
            }
            assert(x % 2 == 0)
            x = x / 2
            j += 1
          }
          assert(x == 0)
          var b = initAgg
          ab.result().reverseIterator.foreach { path =>
            def readToBytes(is: DataInputStream): Array[Byte] = {
              val len = is.readInt()
              val b = new Array[Byte](len)
              is.readFully(b)
              b
            }

            b = combOpFNeedsPool(() =>
              (ctx.r.pool, theHailClassLoaderForSparkWorkers, SparkTaskContext.get())
            )(b, using(new DataInputStream(fsBc.value.open(path)))(readToBytes))
          }
          b
        }

        val aggRegion = ctx.freshRegion()
        val hcl = theHailClassLoaderForSparkWorkers
        val tc = SparkTaskContext.get()
        val newRow = f(hcl, fsBc.value, tc, globalRegion)
        val seq = eltSeqF(hcl, fsBc.value, tc, globalRegion)
        var aggOff = read(hcl, tc, aggRegion, partitionAggs)

        val res = it.map { ptr =>
          newRow.setAggState(aggRegion, aggOff)
          val newPtr = newRow(ctx.region, globals, ptr)
          aggOff = newRow.getAggOffset()
          seq.setAggState(aggRegion, aggOff)
          seq(ctx.region, globals, ptr)
          aggOff = seq.getAggOffset()
          newPtr
        }
        res
      }
      copy(
        typ = newType,
        rvd = rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key))(itF),
      )
    }

    // 2. load in init op on each partition, seq op over partition, write out.
    val scanPartitionAggs = SpillingCollectIterator(
      ctx.localTmpdir,
      ctx.fs,
      rvd.mapPartitionsWithIndex { (i, ctx, it) =>
        val globalRegion = ctx.partitionRegion
        val globals = if (scanSeqNeedsGlobals)
          globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
        else 0

        SparkTaskContext.get().getRegionPool().scopedSmallRegion { aggRegion =>
          val hcl = theHailClassLoaderForSparkWorkers
          val tc = SparkTaskContext.get()
          val seq = eltSeqF(hcl, fsBc.value, tc, globalRegion)

          seq.setAggState(aggRegion, read(hcl, tc, aggRegion, initAgg))
          it.foreach { ptr =>
            seq(ctx.region, globals, ptr)
            ctx.region.clear()
          }
          Iterator.single(write(hcl, tc, aggRegion, seq.getAggOffset()))
        }
      },
      ctx.getFlag("max_leader_scans").toInt,
    )

    // 3. load in partition aggregations, comb op as necessary, write back out.
    val partAggs = scanPartitionAggs.scanLeft(initAgg)(combOpFNeedsPool(() =>
      (ctx.r.pool, ctx.theHailClassLoader, ctx.taskContext)
    ))
    val scanAggCount = rvd.getNumPartitions
    val partitionIndices = new Array[Long](scanAggCount)
    val scanAggsPerPartitionFile = ctx.createTmpPath("table-map-rows-scan-aggs-part")
    using(ctx.fs.createNoCompression(scanAggsPerPartitionFile)) { os =>
      partAggs.zipWithIndex.foreach { case (x, i) =>
        if (i < scanAggCount) {
          log.info(s"TableMapRows scan: serializing combined agg $i")
          partitionIndices(i) = os.getPosition
          os.writeInt(x.length)
          os.write(x, 0, x.length)
        }
      }
    }

    // 4. load in partStarts, calculate newRow based on those results.
    val itF = { (i: Int, ctx: RVDContext, filePosition: Long, it: Iterator[Long]) =>
      val globalRegion = ctx.partitionRegion
      val globals = if (rowIterationNeedsGlobals || scanSeqNeedsGlobals)
        globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
      else
        0
      val partitionAggs = using(fsBc.value.openNoCompression(scanAggsPerPartitionFile)) { is =>
        is.seek(filePosition)
        val aggSize = is.readInt()
        val partAggs = new Array[Byte](aggSize)
        var nread = is.read(partAggs, 0, aggSize)
        var r = nread
        while (r > 0 && nread < aggSize) {
          r = is.read(partAggs, nread, aggSize - nread)
          if (r > 0) nread += r
        }
        if (nread != aggSize) {
          fatal(s"aggs read wrong number of bytes: $nread vs $aggSize")
        }
        partAggs
      }

      val aggRegion = ctx.freshRegion()
      val hcl = theHailClassLoaderForSparkWorkers
      val tc = SparkTaskContext.get()
      val newRow = f(hcl, fsBc.value, tc, globalRegion)
      val seq = eltSeqF(hcl, fsBc.value, tc, globalRegion)
      var aggOff = read(hcl, tc, aggRegion, partitionAggs)

      var idx = 0
      it.map { ptr =>
        newRow.setAggState(aggRegion, aggOff)
        val off = newRow(ctx.region, globals, ptr)
        seq.setAggState(aggRegion, newRow.getAggOffset())
        idx += 1
        seq(ctx.region, globals, ptr)
        aggOff = seq.getAggOffset()
        off
      }
    }

    copy(
      typ = newType,
      rvd = rvd.mapPartitionsWithIndexAndValue(
        RVDType(rTyp.asInstanceOf[PStruct], typ.key),
        partitionIndices,
      )(itF),
    )
  }

  def orderBy(sortFields: IndexedSeq[SortField]): TableValue = {
    val newType = typ.copy(key = FastSeq())
    val physicalKey = rvd.typ.key

    if (TableOrderBy.isAlreadyOrdered(sortFields, physicalKey))
      return copy(typ = newType)

    val rowType = typ.rowType
    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.ordering(ctx.stateManager)
      if (so == Ascending) fo else fo.reverse
    }.toArray

    val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

    val act = implicitly[ClassTag[Annotation]]

    val codec = TypedCodecSpec(ctx, rvd.rowPType, BufferSpec.wireSpec)
    val rdd = rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(_._1)(ord, act)
    val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))
    TableValue(ctx, newType, globals, RVD.unkeyed(rowPType, orderedCRDD))
  }

  def repartition(n: Int, strategy: Int): TableValue =
    copy(rvd = strategy match {
      case RepartitionStrategy.SHUFFLE => rvd.coalesce(ctx, n, shuffle = true)
      case RepartitionStrategy.COALESCE => rvd.coalesce(ctx, n, shuffle = false)
      case RepartitionStrategy.NAIVE_COALESCE => rvd.naiveCoalesce(n, ctx)
    })
}
