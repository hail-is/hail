package is.hail.expr.ir

import cats.mtl.Ask
import cats.syntax.all._
import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.backend.{ExecuteContext, HailStateManager, HailTaskContext, TaskFinalizer}
import is.hail.expr.ir
import is.hail.expr.ir.functions.{BlockMatrixToTableFunction, IntervalFunctions, MatrixToTableFunction, TableToTableFunction}
import is.hail.expr.ir.lowering._
import is.hail.expr.ir.lowering.utils._
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io._
import is.hail.io.avro.AvroTableReader
import is.hail.io.fs.FS
import is.hail.io.index.StagedIndexReader
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, BlockMatrixReadRowBlockedRDD}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SInt64, SInt64Value}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.prettyPrint.ArrayOfByteArrayInputStream
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import java.io.{Closeable, DataInputStream, DataOutputStream, InputStream}
import scala.language.higherKinds


object TableIR {
  def read(fs: FS, path: String, dropRows: Boolean = false, requestedType: Option[TableType] = None): TableRead = {
    val successFile = path + "/_SUCCESS"
    if (!fs.exists(path + "/_SUCCESS"))
      fatal(s"write failed: file not found: $successFile")

    val tr = TableNativeReader.read(fs, path, None)
    TableRead(requestedType.getOrElse(tr.fullType), dropRows = dropRows, tr)
  }
}

abstract sealed class TableIR extends BaseIR {
  def typ: TableType

  def partitionCounts: Option[IndexedSeq[Long]] = None

  val rowCountUpperBound: Option[Long]

  final def analyzeAndExecute[M[_]: MonadLower]: M[TableExecuteIntermediate] =
    LoweringAnalyses(this) >>= execute[M]

  protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    prettyFatal(pretty => "tried to execute unexecutable IR:\n" + pretty(this))

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR

  def unpersist(): TableIR = {
    this match {
      case TableLiteral(typ, rvd, enc, encodedGlobals) => TableLiteral(typ, rvd.unpersist(), enc, encodedGlobals)
      case x => x
    }
  }

  def pyUnpersist(): TableIR = unpersist()
}

object TableLiteral {
  def apply[M[_]](value: TableValue)(implicit M: Ask[M, ExecuteContext]): M[TableLiteral] =
    M.applicative.map(value.globals.encodeToByteArrays) { bytes =>
      TableLiteral(value.typ, value.rvd, value.globals.encoding, bytes)
    }
}

case class TableLiteral(typ: TableType, rvd: RVD, enc: AbstractTypedCodecSpec, encodedGlobals: Array[Array[Byte]]) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    TableLiteral(typ, rvd, enc, encodedGlobals)
  }


  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M])
  : M[TableExecuteIntermediate] =
    M.reader { ctx =>
      val (globalPType: PStruct, dec) = enc.buildDecoder(ctx, typ.globalType)
      val bais = new ArrayOfByteArrayInputStream(encodedGlobals)
      val globalOffset = dec.apply(bais, ctx.theHailClassLoader).readRegionValue(ctx.r)
      TableValueIntermediate(TableValue(typ, BroadcastRow(ctx.stateManager, RegionValue(ctx.r, globalOffset), globalPType), rvd))
    }
}

object TableReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[TableNativeZippedReader])
  ) + new NativeReaderOptionsSerializer()

  def fromJValue(fs: FS, jv: JValue): TableReader = {
    (jv \ "name").extract[String] match {
      case "TableNativeReader" => TableNativeReader.fromJValue(fs, jv)
      case "TableFromBlockMatrixNativeReader" => TableFromBlockMatrixNativeReader.fromJValue(fs, jv)
      case "StringTableReader" => StringTableReader.fromJValue(fs, jv)
      case "AvroTableReader" => AvroTableReader.fromJValue(jv)
      case _ => jv.extract[TableReader]
    }
  }

  val uidFieldName = "__row_uid"
}

object LoweredTableReader {

  private[this] val coercerCache: Cache[Any, LoweredTableReaderCoercer] = new Cache(32)

  def makeCoercer[M[_]](
    key: IndexedSeq[String],
    partitionKey: Int,
    uidFieldName: String,
    contextType: TStruct,
    contexts: IndexedSeq[Any],
    keyType: TStruct,
    bodyPType: TStruct => PStruct,
    keys: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long],
    context: String,
    cacheKey: Any
  )(implicit M: MonadLower[M]): M[LoweredTableReaderCoercer] = {
    assert(contexts.nonEmpty)
    assert(contextType.hasField("partitionIndex"))
    assert(contextType.fieldType("partitionIndex") == TInt32)

    val cacheKeyWithInfo = (partitionKey, keyType, key, cacheKey)
    coercerCache.get(cacheKeyWithInfo) match {
      case Some(r) => M.pure(r)
      case None =>
        info(s"scanning $context for sortedness...")
        val nPartitions = contexts.length
        val sampleSize = math.min(nPartitions * 20, 1000000)
        val samplesPerPartition = sampleSize / nPartitions

        val pkType = keyType.typeAfterSelectNames(key.take(partitionKey))

        def selectPK(k: IR): IR =
          SelectFields(k, key.take(partitionKey))

        val prevkey = AggSignature(PrevNonnull(),
          FastIndexedSeq(),
          FastIndexedSeq(keyType))

        val count = AggSignature(Count(),
          FastIndexedSeq(),
          FastIndexedSeq())

        val xType = TStruct(
          "key" -> keyType,
          "token" -> TFloat64,
          "prevkey" -> keyType)

        val samplekey = AggSignature(TakeBy(),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, TFloat64))

        val sum = AggSignature(Sum(),
          FastIndexedSeq(),
          FastIndexedSeq(TInt64))

        val minkey = AggSignature(TakeBy(),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))

        val maxkey = AggSignature(TakeBy(Descending),
          FastIndexedSeq(TInt32),
          FastIndexedSeq(keyType, keyType))

        val scanBody = (ctx: IR) => StreamAgg(
          StreamAggScan(
            ReadPartition(ctx, keyType, new PartitionIteratorLongReader(
              keyType,
              uidFieldName,
              contextType,
              (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
              (requestedType: Type) => keys(requestedType.asInstanceOf[TStruct]))),
            "key",
            MakeStruct(FastIndexedSeq(
              "key" -> Ref("key", keyType),
              "token" -> invokeSeeded("rand_unif", 1, TFloat64, RNGStateLiteral(), F64(0.0), F64(1.0)),
              "prevkey" -> ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("key", keyType)), prevkey)))),
          "x",
          Let("n", ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(), count),
            AggLet("key", GetField(Ref("x", xType), "key"),
              MakeStruct(FastIndexedSeq(
                "n" -> Ref("n", TInt64),
                "minkey" ->
                  ApplyAggOp(
                    FastIndexedSeq(I32(1)),
                    FastIndexedSeq(Ref("key", keyType), Ref("key", keyType)),
                    minkey),
                "maxkey" ->
                  ApplyAggOp(
                    FastIndexedSeq(I32(1)),
                    FastIndexedSeq(Ref("key", keyType), Ref("key", keyType)),
                    maxkey),
                "ksorted" ->
                  ApplyComparisonOp(EQ(TInt64),
                    ApplyAggOp(
                      FastIndexedSeq(),
                      FastIndexedSeq(
                        invoke("toInt64", TInt64,
                          invoke("lor", TBoolean,
                            IsNA(GetField(Ref("x", xType), "prevkey")),
                            ApplyComparisonOp(LTEQ(keyType),
                              GetField(Ref("x", xType), "prevkey"),
                              GetField(Ref("x", xType), "key"))))),
                      sum),
                    Ref("n", TInt64)),
                "pksorted" ->
                  ApplyComparisonOp(EQ(TInt64),
                    ApplyAggOp(
                      FastIndexedSeq(),
                      FastIndexedSeq(
                        invoke("toInt64", TInt64,
                          invoke("lor", TBoolean,
                            IsNA(selectPK(GetField(Ref("x", xType), "prevkey"))),
                            ApplyComparisonOp(LTEQ(pkType),
                              selectPK(GetField(Ref("x", xType), "prevkey")),
                              selectPK(GetField(Ref("x", xType), "key")))))),
                      sum),
                    Ref("n", TInt64)),
                "sample" -> ApplyAggOp(
                  FastIndexedSeq(I32(samplesPerPartition)),
                  FastIndexedSeq(GetField(Ref("x", xType), "key"), GetField(Ref("x", xType), "token")),
                  samplekey))),
              isScan = false)))

        val scanResult = CollectDistributedArray(
          ToStream(Literal(TArray(contextType), contexts)),
          MakeStruct(FastIndexedSeq()),
          "context",
          "globals",
          scanBody(Ref("context", contextType)),
          NA(TString),
          "table_coerce_sortedness"
        )

        val sortedPartDataIR = sortIR(bindIR(scanResult) { scanResult =>
          mapIR(
            filterIR(
              mapIR(
                rangeIR(I32(0), ArrayLen(scanResult))) { i =>
                InsertFields(
                  ArrayRef(scanResult, i),
                  FastIndexedSeq("i" -> i))
              }) { row => ArrayLen(GetField(row, "minkey")) > 0 }
          ) { row =>
            InsertFields(row, FastSeq(
              ("minkey", ArrayRef(GetField(row, "minkey"), I32(0))),
              ("maxkey", ArrayRef(GetField(row, "maxkey"), I32(0)))))
          }
        }) { (l, r) =>
          ApplyComparisonOp(LT(TStruct("minkey" -> keyType, "maxkey" -> keyType)),
            SelectFields(l, FastSeq("minkey", "maxkey")),
            SelectFields(r, FastSeq("minkey", "maxkey")))
        }

        val partDataElt = tcoerce[TArray](sortedPartDataIR.typ).elementType

        val summary =
          Let("sortedPartData", sortedPartDataIR,
            MakeStruct(FastIndexedSeq(
              "ksorted" ->
                invoke("land", TBoolean,
                  StreamFold(ToStream(Ref("sortedPartData", sortedPartDataIR.typ)),
                    True(),
                    "acc",
                    "partDataWithIndex",
                    invoke("land", TBoolean,
                      Ref("acc", TBoolean),
                      GetField(Ref("partDataWithIndex", partDataElt), "ksorted"))),
                  StreamFold(
                    StreamRange(
                      I32(0),
                      ArrayLen(Ref("sortedPartData", sortedPartDataIR.typ)) - I32(1),
                      I32(1)),
                    True(),
                    "acc", "i",
                    invoke("land", TBoolean,
                      Ref("acc", TBoolean),
                      ApplyComparisonOp(LTEQ(keyType),
                        GetField(
                          ArrayRef(Ref("sortedPartData", sortedPartDataIR.typ), Ref("i", TInt32)),
                          "maxkey"),
                        GetField(
                          ArrayRef(Ref("sortedPartData", sortedPartDataIR.typ), Ref("i", TInt32) + I32(1)),
                          "minkey"))))),
              "pksorted" ->
                invoke("land", TBoolean,
                  StreamFold(ToStream(Ref("sortedPartData", sortedPartDataIR.typ)),
                    True(),
                    "acc",
                    "partDataWithIndex",
                    invoke("land", TBoolean,
                      Ref("acc", TBoolean),
                      GetField(Ref("partDataWithIndex", partDataElt), "pksorted"))),
                  StreamFold(
                    StreamRange(
                      I32(0),
                      ArrayLen(Ref("sortedPartData", sortedPartDataIR.typ)) - I32(1),
                      I32(1)),
                    True(),
                    "acc", "i",
                    invoke("land", TBoolean,
                      Ref("acc", TBoolean),
                      ApplyComparisonOp(LTEQ(pkType),
                        selectPK(GetField(
                          ArrayRef(Ref("sortedPartData", sortedPartDataIR.typ), Ref("i", TInt32)),
                          "maxkey")),
                        selectPK(GetField(
                          ArrayRef(Ref("sortedPartData", sortedPartDataIR.typ), Ref("i", TInt32) + I32(1)),
                          "minkey")))))),
              "sortedPartData" -> Ref("sortedPartData", sortedPartDataIR.typ))))

        for {
          (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) <-
            Compile[M, AsmFunction1RegionLong](
              FastIndexedSeq(),
              FastIndexedSeq[TypeInfo[_]](classInfo[Region]),
              LongInfo,
              summary,
              optimize = true
            )

          row <- scopedExecution { case (hcl, fs, htc, r) =>
            M.pure(SafeRow(resultPType, f(hcl, fs, htc, r)(r)))
          }

          ksorted = row.getBoolean(0)
          pksorted = row.getBoolean(1)
          sortedPartData = row.getAs[IndexedSeq[Row]](2)

          coercer = if (ksorted) {
            info(s"Coerced sorted ${context} - no additional import work to do")
            new LoweredTableReaderCoercer {
              override def apply[F[_]: MonadLower](
                globals: IR,
                contextType: Type,
                contexts: IndexedSeq[Any],
                body: IR => IR
              ): F[TableStage] =
                MonadLower[F].reader { ctx =>
                  val partOrigIndex = sortedPartData.map(_.getInt(6))

                  val partitioner = new RVDPartitioner(ctx.stateManager, keyType,
                    sortedPartData.map { partData =>
                      Interval(partData.get(1), partData.get(2), includesStart = true, includesEnd = true)
                    },
                    key.length
                  )

                  TableStage(globals, partitioner, TableStageDependency.none,
                    ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
                    body
                  )
                }
            }
          } else if (pksorted) {
            info(s"Coerced prefix-sorted $context, requiring additional sorting within data partitions on each query.")

            new LoweredTableReaderCoercer {
              private[this] def selectPK(r: Row): Row = {
                val a = new Array[Any](partitionKey)
                var i = 0
                while (i < partitionKey) {
                  a(i) = r.get(i)
                  i += 1
                }
                Row.fromSeq(a)
              }

              override def apply[F[_]](globals: IR,
                                       contextType: Type,
                                       contexts: IndexedSeq[Any],
                                       body: IR => IR
                                      )
                                      (implicit F: MonadLower[F])
              : F[TableStage] =
                F.ask.flatMap { ctx =>
                  val partOrigIndex = sortedPartData.map(_.getInt(6))

                  val partitioner =
                    new RVDPartitioner(ctx.stateManager, pkType,
                      sortedPartData.map { partData =>
                        Interval(selectPK(partData.getAs[Row](1)), selectPK(partData.getAs[Row](2)), includesStart = true, includesEnd = true)
                      }, pkType.size
                    )

                  val pkPartitioned =
                    TableStage(globals, partitioner, TableStageDependency.none,
                      ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
                      body
                    )

                  pkPartitioned.extendKeyPreservesPartitioning[F](key).map {
                    _.mapPartition(None) { part =>
                      flatMapIR(StreamGroupByKey(part, pkType.fieldNames, missingEqual = true)) { inner =>
                        ToStream(sortIR(inner) { case (l, r) => ApplyComparisonOp(LT(l.typ), l, r) })
                      }
                    }
                  }
                }
            }
          } else {
            info(s"$context is out of order..." +
              s"\n  Write the dataset to disk before running multiple queries to avoid multiple costly data shuffles.")

            new LoweredTableReaderCoercer {
              override def apply[F[_]](globals: IR,
                                       contextType: Type,
                                       contexts: IndexedSeq[Any],
                                       body: IR => IR
                                       )
                                      (implicit F: MonadLower[F])
              : F[TableStage] =
                for {
                  ctx <- F.ask
                  partOrigIndex = sortedPartData.map(_.getInt(6))
                  partitioner = RVDPartitioner.unkeyed(ctx.stateManager, sortedPartData.length)
                  tableStage = TableStage(globals, partitioner, TableStageDependency.none,
                    ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
                    body
                  )

                  rowRType = VirtualTypeWithReq(bodyPType(tableStage.rowType)).r.asInstanceOf[RStruct]
                  globReq <- Requiredness[F](globals)
                  globRType = globReq.lookup(globals).asInstanceOf[RStruct]

                  sorted <- ctx.backend.lowerDistributedSort[F](
                    tableStage,
                    keyType.fieldNames.map(f => SortField(f, Ascending)),
                    RTable(rowRType, globRType, FastSeq()),
                  )

                  ts <- sorted.lower[F](
                    TableType(tableStage.rowType, keyType.fieldNames, globals.typ.asInstanceOf[TStruct])
                  )
                } yield ts
            }
          }

          _ <- M.reader(_.backend.shouldCacheQueryInfo)
            .ifF(M.pure { coercerCache += (cacheKeyWithInfo -> coercer) }, M.unit)
        } yield coercer
    }
  }
}

trait TableReaderWithExtraUID extends TableReader {

  def fullTypeWithoutUIDs: TableType

  final val uidFieldName = TableReader.uidFieldName

  lazy val fullType: TableType = {
    require(!fullTypeWithoutUIDs.rowType.hasField(uidFieldName))
    fullTypeWithoutUIDs.copy(
      rowType = fullTypeWithoutUIDs.rowType.insertFields(
        Array((uidFieldName, uidType))))
  }

  def uidType: Type


  protected def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq

  protected def uidRequiredness: VirtualTypeWithReq

  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    val requestedUID = requestedType.rowType.hasField(uidFieldName)
    val concreteRowType = if (requestedUID)
      requestedType.rowType.deleteKey(uidFieldName)
    else
      requestedType.rowType
    val concreteRowReq = concreteRowRequiredness(ctx, requestedType.copy(rowType = concreteRowType))
    if (requestedUID) {
      val concreteRFields = concreteRowReq.r.asInstanceOf[RStruct].fields
      VirtualTypeWithReq(
        requestedType.rowType,
        RStruct(concreteRFields :+ RField(uidFieldName, uidRequiredness.r, concreteRFields.length)))
    } else {
      concreteRowReq
    }
  }
}
abstract class TableReader {
  def pathsUsed: Seq[String]

  def apply[M[_]: MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue]

  def partitionCounts: Option[IndexedSeq[Long]]

  def isDistinctlyKeyed: Boolean = false // FIXME: No default value

  def fullType: TableType

  def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq

  def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq

  def toJValue: JValue = {
    Extraction.decompose(this)(TableReader.formats)
  }

  def renderShort(): String

  def defaultRender(): String = {
    StringEscapeUtils.escapeString(JsonMethods.compact(toJValue))
  }

  def lowerGlobals[M[_]](requestedGlobalsType: TStruct)(implicit M: MonadLower[M]): M[IR] =
    M.raiseError(new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lowerGlobals not implemented"))

  def lower[M[_]](requestedType: TableType)(implicit M: MonadLower[M]): M[TableStage] =
    M.raiseError(new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lower not implemented"))
}

object TableNativeReader {
  def read(fs: FS, path: String, options: Option[NativeReaderOptions]): TableNativeReader =
    TableNativeReader(fs, TableNativeReaderParameters(path, options))

  def apply(fs: FS, params: TableNativeReaderParameters): TableNativeReader = {
    val spec = (RelationalSpec.read(fs, params.path): @unchecked) match {
      case ts: AbstractTableSpec => ts
      case _: AbstractMatrixTableSpec => fatal(s"file is a MatrixTable, not a Table: '${ params.path }'")
    }

    val filterIntervals = params.options.map(_.filterIntervals).getOrElse(false)

    if (filterIntervals && !spec.indexed)
      fatal(
        """`intervals` specified on an unindexed table.
          |This table was written using an older version of hail
          |rewrite the table in order to create an index to proceed""".stripMargin)

    new TableNativeReader(params, spec)
  }

  def fromJValue(fs: FS, jv: JValue): TableNativeReader = {
    implicit val formats: Formats = DefaultFormats + new NativeReaderOptionsSerializer()
    val params = jv.extract[TableNativeReaderParameters]
    TableNativeReader(fs, params)
  }
}


case class PartitionRVDReader(rvd: RVD, uidFieldName: String) extends PartitionReader {
  override def contextType: Type = TInt32

  override def fullRowType: TStruct = rvd.rowType.insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64)))

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val tr = TypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    tr.fromPType(rvd.rowPType)
    tr
  }

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    import Lower.monadLowerInstanceForLower
    val (Some(PTypeReferenceSingleCodeType(upcastPType: PBaseStruct)), upcast) =
      Compile[Lower, AsmFunction2RegionLongLong](
        FastIndexedSeq(("elt", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)))),
        FastIndexedSeq(classInfo[Region], LongInfo),
        LongInfo,
        PruneDeadFields.upcast(ctx, Ref("elt", rvd.rowType), requestedType)
      )
        .runA(ctx, LoweringState())

    val upcastCode = mb.getObject[Function4[HailClassLoader, FS, HailTaskContext, Region, AsmFunction2RegionLongLong]](upcast)

    val rowPType = rvd.rowPType.subsetTo(requestedType)

    val createUID = requestedType.hasField(uidFieldName)

    assert(upcastPType == rowPType,
      s"ptype mismatch:\n  upcast: $upcastPType\n  computed: ${rowPType}\n  inputType: ${rvd.rowPType}\n  requested: ${requestedType}")

    context.toI(cb).map(cb) { _partIdx =>
      val partIdx = cb.memoizeField(_partIdx, "partIdx")
      val iterator = mb.genFieldThisRef[Iterator[Long]]("rvdreader_iterator")
      val next = mb.genFieldThisRef[Long]("rvdreader_next")
      val curIdx = mb.genFieldThisRef[Long]("rvdreader_curIdx")

      val region = mb.genFieldThisRef[Region]("rvdreader_region")
      val upcastF = mb.genFieldThisRef[AsmFunction2RegionLongLong]("rvdreader_upcast")

      val broadcastRVD = mb.getObject[BroadcastRVD](new BroadcastRVD(ctx.backend.asSpark("RVDReader"), rvd))

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb

        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          cb.assign(curIdx, 0L)
          cb.assign(iterator, broadcastRVD.invoke[Int, Region, Region, Iterator[Long]](
            "computePartition", partIdx.asInt.value, region, partitionRegion))
          cb.assign(upcastF, Code.checkcast[AsmFunction2RegionLongLong](upcastCode.invoke[AnyRef, AnyRef, AnyRef, AnyRef, AnyRef](
            "apply", cb.emb.ecb.emodb.getHailClassLoader, cb.emb.ecb.emodb.getFS, cb.emb.ecb.getTaskContext, partitionRegion)))
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!iterator.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(curIdx, curIdx + 1)
          cb.assign(next, upcastF.invoke[Region, Long, Long]("apply", region, Code.longValue(iterator.invoke[java.lang.Long]("next"))))
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          if (createUID) {
            val uid = SStackStruct.constructFromArgs(cb, region, TTuple(TInt64, TInt64),
              EmitCode.present(mb, partIdx), EmitCode.present(mb, primitive(cb.memoize(curIdx - 1))))
            IEmitCode.present(cb, upcastPType.loadCheapSCode(cb, next)
              ._insert(requestedType, uidFieldName -> EmitValue.present(uid)))
          } else {
            IEmitCode.present(cb, upcastPType.loadCheapSCode(cb, next))
          }
        }

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamValue(producer)
    }
  }

  def toJValue: JValue = JString("<PartitionRVDReader>") // cannot be parsed, but need a printout for Pretty
}

trait AbstractNativeReader extends PartitionReader {
  def uidFieldName: String

  def spec: AbstractTypedCodecSpec

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val tr = TypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    val pType = if (requestedType.hasField(uidFieldName)) {
      val basePType = spec.decodedPType(requestedType.deleteKey(uidFieldName)).asInstanceOf[PStruct]
      val uidPType = PCanonicalTuple(true, PInt64Required, PInt64Required)
      basePType.insertFields(Array(uidFieldName -> uidPType))
    } else {
      spec.decodedPType(requestedType)
    }
    tr.fromPType(pType)
    tr
  }

  def fullRowType: TStruct = spec.encodedVirtualType.asInstanceOf[TStruct]
    .insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64)))
}

case class PartitionNativeReader(spec: AbstractTypedCodecSpec, uidFieldName: String)
  extends AbstractNativeReader {

  def contextType: Type = TStruct("partitionIndex" -> TInt64, "partitionPath" -> TString)

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName) && !spec.encodedVirtualType.asInstanceOf[TStruct].hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType

    val concreteSType = spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStruct]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)))
    val elementSType = if (insertUID)
      SInsertFieldsStruct(requestedType, concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)))
    else
      concreteSType

    context.toI(cb).map(cb) { case ctxStruct: SBaseStructValue =>
      val partIdx = cb.memoizeField(ctxStruct.loadField(cb, "partitionIndex").get(cb), "partIdx")
      val rowIdx = mb.genFieldThisRef[Long]("pnr_rowidx")
      val pathString = cb.memoizeField(ctxStruct.loadField(cb, "partitionPath").get(cb).asString.loadString(cb))
      val xRowBuf = mb.genFieldThisRef[InputBuffer]("pnr_xrowbuf")
      val next = mb.newPSettable(mb.fieldBuilder, elementSType, "pnr_next")
      val region = mb.genFieldThisRef[Region]("pnr_region")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          cb.assign(xRowBuf, spec.buildCodeInputBuffer(mb.openUnbuffered(pathString, checkCodec = true)))
          cb.assign(rowIdx, -1L)
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!xRowBuf.readByte().toZ, cb.goto(LendOfStream))

          val base = spec.encodedType.buildDecoder(concreteType, cb.emb.ecb).apply(cb, region, xRowBuf).asBaseStruct
          if (insertUID) {
            cb.assign(rowIdx, rowIdx + 1)
            val uid = EmitValue.present(
              new SStackStructValue(uidSType, Array(
                EmitValue.present(partIdx),
                EmitValue.present(new SInt64Value(rowIdx)))))
            cb.assign(next, base._insert(requestedType, uidFieldName -> uid))
          } else
            cb.assign(next, base)

          cb.goto(LproduceElementDone)
        }

        override val element: EmitCode = EmitCode.present(mb, next)

        override def close(cb: EmitCodeBuilder): Unit = cb += xRowBuf.close()
      }
      SStreamValue(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class PartitionNativeIntervalReader(sm: HailStateManager, tablePath: String, tableSpec: AbstractTableSpec, uidFieldName: String) extends AbstractNativeReader {
  require(tableSpec.indexed)

  lazy val rowsSpec = tableSpec.rowsSpec
  lazy val spec = rowsSpec.typedCodecSpec
  lazy val indexSpec = tableSpec.rowsSpec.asInstanceOf[Indexed].indexSpec
  lazy val partitioner = rowsSpec.partitioner(sm)

  lazy val contextType: Type = RVDPartitioner.intervalIRRepresentation(partitioner.kType)

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType
    val concreteSType: SBaseStruct = spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStruct]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)))
    val eltSType: SBaseStruct = if (insertUID)
      SInsertFieldsStruct(requestedType, concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)))
    else
      concreteSType

    val index = new StagedIndexReader(cb.emb, indexSpec.leafCodec, indexSpec.internalNodeCodec)

    context.toI(cb).map(cb) { case _ctx: SIntervalValue =>
      val ctx = cb.memoizeField(_ctx, "ctx").asInterval

      val partitionerLit = partitioner.partitionBoundsIRRepresentation
      val partitionerRuntime = cb.emb.addLiteral(cb, partitionerLit.value, VirtualTypeWithReq.fullyOptional(partitionerLit.typ))
        .asIndexable

      val pathsType = VirtualTypeWithReq.fullyRequired(TArray(TString))
      val rowsPath = tableSpec.rowsComponent.absolutePath(tablePath)
      val partitionPathsRuntime = cb.memoizeField(mb.addLiteral(cb, rowsSpec.absolutePartPaths(rowsPath).toFastIndexedSeq, pathsType), "partitionPathsRuntime")
        .asIndexable
      val indexPathsRuntime = cb.memoizeField(mb.addLiteral(cb, rowsSpec.partFiles.map(partPath => s"${ rowsPath }/${ indexSpec.relPath }/${ partPath }.idx").toFastIndexedSeq, pathsType), "indexPathsRuntime")
        .asIndexable

      val currIdxInPartition = mb.genFieldThisRef[Long]("n_to_read")
      val stopIdxInPartition = mb.genFieldThisRef[Long]("n_to_read")
      val finalizer = mb.genFieldThisRef[TaskFinalizer]("finalizer")

      val startPartitionIndex = mb.genFieldThisRef[Int]("start_part")
      val currPartitionIdx = mb.genFieldThisRef[Int]("curr_part")
      val lastIncludedPartitionIdx = mb.genFieldThisRef[Int]("last_part")
      val ib = mb.genFieldThisRef[InputBuffer]("buffer")

      // leave the index open/initialized to allow queries to reuse the same index for the same file
      val indexInitialized = mb.genFieldThisRef[Boolean]("index_init")
      val indexCachedIndex = mb.genFieldThisRef[Int]("index_last_idx")
      val streamFirst = mb.genFieldThisRef[Boolean]("stream_first")

      val region = mb.genFieldThisRef[Region]("pnr_region")

      val decodedRow = cb.emb.newPField("rowsValue", eltSType)

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {

          val startBound = ctx.loadStart(cb).get(cb)
          val includesStart = ctx.includesStart
          val endBound = ctx.loadEnd(cb).get(cb)
          val includesEnd = ctx.includesEnd

          val (startPart, endPart) = IntervalFunctions.partitionerFindIntervalRange(cb,
            partitionerRuntime,
            SStackInterval.construct(EmitValue.present(startBound), EmitValue.present(endBound), includesStart, includesEnd),
            -1)

          cb.ifx(endPart < startPart, cb._fatal("invalid start/end config - startPartIdx=",
            startPartitionIndex.toS, ", endPartIdx=", lastIncludedPartitionIdx.toS))

          cb.assign(startPartitionIndex, startPart)
          cb.assign(lastIncludedPartitionIdx, endPart - 1)
          cb.assign(currPartitionIdx, startPartitionIndex)


          cb.assign(streamFirst, true)
          cb.assign(currIdxInPartition, 0L)
          cb.assign(stopIdxInPartition, 0L)

          cb.assign(finalizer, cb.emb.ecb.getTaskContext.invoke[TaskFinalizer]("newFinalizer"))
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          val Lstart = CodeLabel()
          cb.define(Lstart)
          cb.ifx(currIdxInPartition >= stopIdxInPartition, {
            cb.ifx(currPartitionIdx >= partitioner.numPartitions || currPartitionIdx > lastIncludedPartitionIdx,
              cb.goto(LendOfStream))

            val requiresIndexInit = cb.newLocal[Boolean]("requiresIndexInit")

            cb.ifx(streamFirst, {
              // if first, reuse open index from previous time the stream was run if possible
              // this is a common case if looking up nearby keys
              cb.assign(requiresIndexInit, !(indexInitialized && (indexCachedIndex ceq currPartitionIdx)))
            }, {
              // if not first, then the index must be open to the previous partition and needs to be reinitialized
              cb.assign(streamFirst, false)
              cb.assign(requiresIndexInit, true)
            })

            cb.ifx(requiresIndexInit, {
              cb.ifx(indexInitialized, {
                cb += finalizer.invoke[Unit]("clear")
                index.close(cb)
                cb += ib.close()
              }, {
                cb.assign(indexInitialized, true)
              })
              cb.assign(indexCachedIndex, currPartitionIdx)
              val partPath = partitionPathsRuntime.loadElement(cb, currPartitionIdx).get(cb).asString.loadString(cb)
              val idxPath = indexPathsRuntime.loadElement(cb, currPartitionIdx).get(cb).asString.loadString(cb)
              index.initialize(cb, idxPath)
              cb.assign(ib, spec.buildCodeInputBuffer(
                Code.newInstance[ByteTrackingInputStream, InputStream](
                  cb.emb.openUnbuffered(partPath, false))))
              index.addToFinalizer(cb, finalizer)
              cb += finalizer.invoke[Closeable, Unit]("addCloseable", ib)
            })

            cb.ifx(currPartitionIdx ceq lastIncludedPartitionIdx, {
              cb.ifx(currPartitionIdx ceq startPartitionIndex, {
                // query the full interval
                val indexResult = index.queryInterval(cb, ctx)
                val startIdx = indexResult.loadField(cb, 0)
                  .get(cb)
                  .asInt64
                  .value
                cb.assign(currIdxInPartition, startIdx)
                val endIdx = indexResult.loadField(cb, 1)
                  .get(cb)
                  .asInt64
                  .value
                cb.assign(stopIdxInPartition, endIdx)
                cb.ifx(endIdx > startIdx, {
                  val firstOffset = indexResult.loadField(cb, 2)
                    .get(cb)
                    .asBaseStruct
                    .loadField(cb, "offset")
                    .get(cb)
                    .asInt64
                    .value

                  cb += ib.seek(firstOffset)
                })
              }, {
                // read from start of partition to the end interval

                val indexResult = index.queryBound(cb, ctx.loadEnd(cb).get(cb).asBaseStruct, ctx.includesEnd)
                val startIdx = indexResult.loadField(cb, 0).get(cb).asInt64.value
                cb.assign(currIdxInPartition, 0L)
                cb.assign(stopIdxInPartition, startIdx)
                // no need to seek, starting at beginning of partition
              })
            }, {
              cb.ifx(currPartitionIdx ceq startPartitionIndex,
                {
                  // read from left endpoint until end of partition
                  val indexResult = index.queryBound(cb, ctx.loadStart(cb).get(cb).asBaseStruct, cb.memoize(!ctx.includesStart))
                  val startIdx = indexResult.loadField(cb, 0).get(cb).asInt64.value

                  cb.assign(currIdxInPartition, startIdx)
                  cb.assign(stopIdxInPartition, index.nKeys(cb))
                  cb.ifx(currIdxInPartition < stopIdxInPartition, {
                    val firstOffset = indexResult.loadField(cb, 1).get(cb).asBaseStruct
                      .loadField(cb, "offset").get(cb).asInt64.value

                    cb += ib.seek(firstOffset)
                  })
                }, {
                  // in the middle of a partition run, so read everything
                  cb.assign(currIdxInPartition, 0L)
                  cb.assign(stopIdxInPartition, index.nKeys(cb))
                })
            })

            cb.assign(currPartitionIdx, currPartitionIdx + 1)
            cb.goto(Lstart)
          })

          cb.ifx(ib.readByte() cne 1, cb._fatal(s"bad buffer state!"))
          cb.assign(currIdxInPartition, currIdxInPartition + 1L)
          val decRow = spec.encodedType.buildDecoder(requestedType, cb.emb.ecb)(cb, region, ib).asBaseStruct
          cb.assign(decodedRow, if (insertUID)
            decRow.insert(cb,
              elementRegion,
              eltSType.virtualType.asInstanceOf[TStruct],
              uidFieldName -> EmitValue.present(uidSType.fromEmitCodes(cb,
                FastIndexedSeq(
                EmitCode.present(mb, primitive(currPartitionIdx)),
                EmitCode.present(mb, primitive(currIdxInPartition))))))
            else decRow)
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          IEmitCode.present(cb, decodedRow)
        }

        override def close(cb: EmitCodeBuilder): Unit = {
          // no cleanup! leave the index open for the next time the stream is run.
          // the task finalizer will clean up the last open index, so this node
          // leaks 2 open file handles until the end of the task.
        }
      }
      SStreamValue(producer)
    }
  }
}

case class PartitionNativeReaderIndexed(
  spec: AbstractTypedCodecSpec,
  indexSpec: AbstractIndexSpec,
  key: IndexedSeq[String],
  uidFieldName: String
) extends AbstractNativeReader {
  def contextType: Type = TStruct(
    "partitionIndex" -> TInt64,
    "partitionPath" -> TString,
    "indexPath" -> TString,
    "interval" -> RVDPartitioner.intervalIRRepresentation(spec.encodedVirtualType.asInstanceOf[TStruct].select(key)._1))

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType
    val concreteSType: SBaseStructPointer = spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStructPointer]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)))
    val eltSType: SBaseStruct = if (insertUID)
      SInsertFieldsStruct(requestedType, concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)))
    else
      concreteSType
    val index = new StagedIndexReader(cb.emb, indexSpec.leafCodec, indexSpec.internalNodeCodec)

    context.toI(cb).map(cb) { case ctxStruct: SBaseStructValue =>
      val partIdx = cb.memoizeField(ctxStruct.loadField(cb, "partitionIndex").get(cb), "partIdx")
      val curIdx = mb.genFieldThisRef[Long]("cur_index")
      val endIdx = mb.genFieldThisRef[Long]("end_index")
      val ib = mb.genFieldThisRef[InputBuffer]("buffer")

      val region = mb.genFieldThisRef[Region]("pnr_region")

      val decodedRow = cb.emb.newPField("rowsValue", eltSType)

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = cb.emb
        override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => (endIdx - curIdx).toI)

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
          val indexPath = ctxStruct
            .loadField(cb, "indexPath")
            .get(cb)
            .asString
            .loadString(cb)
          val partitionPath = ctxStruct
            .loadField(cb, "partitionPath")
            .get(cb)
            .asString
            .loadString(cb)
          val interval = ctxStruct
            .loadField(cb, "interval")
            .get(cb)
            .asInterval
          index.initialize(cb, indexPath)

          val indexResult = index.queryInterval(cb, interval)
          val startIndex = indexResult.loadField(cb, 0)
            .get(cb)
            .asInt64
            .value
          val endIndex = indexResult.loadField(cb, 1)
            .get(cb)
            .asInt64
            .value
          cb.assign(curIdx, startIndex)
          cb.assign(endIdx, endIndex)

          cb.assign(ib, spec.buildCodeInputBuffer(
            Code.newInstance[ByteTrackingInputStream, InputStream](
              cb.emb.openUnbuffered(partitionPath, false))))
          cb.ifx(endIndex > startIndex, {
            val firstOffset = indexResult.loadField(cb, 2)
              .get(cb)
              .asBaseStruct
              .loadField(cb, "offset")
              .get(cb)
              .asInt64
              .value

            cb += ib.seek(firstOffset)
          })
          index.close(cb)
        }
        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(curIdx >= endIdx, cb.goto(LendOfStream))
          val next = ib.readByte()
          cb.ifx(next cne 1, cb._fatal(s"bad buffer state!"))
          val base = spec.encodedType.buildDecoder(concreteType, cb.emb.ecb)(cb, region, ib).asBaseStruct
          if (insertUID)
            cb.assign(decodedRow, new SInsertFieldsStructValue(
              eltSType.asInstanceOf[SInsertFieldsStruct],
              base,
              Array(EmitValue.present(
                new SStackStructValue(uidSType, Array(
                  EmitValue.present(partIdx),
                  EmitValue.present(primitive(curIdx))))))))
          else
            cb.assign(decodedRow, base)
          cb.assign(curIdx, curIdx + 1L)
          cb.goto(LproduceElementDone)

        }
        override val element: EmitCode = EmitCode.present(mb, decodedRow)

        override def close(cb: EmitCodeBuilder): Unit = cb += ib.close()
      }
      SStreamValue(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

// Result uses the uid field name and values from the right input, and ignores
// uids from the left.
case class PartitionZippedNativeReader(left: PartitionReader, right: PartitionReader)
  extends PartitionReader {

  def uidFieldName = right.uidFieldName

  def contextType: Type = TStruct(
    "leftContext" -> left.contextType,
    "rightContext" -> right.contextType)

  def splitRequestedType(requestedType: TStruct): (TStruct, TStruct) = {
    val leftStruct = left.fullRowType.deleteKey(left.uidFieldName)
    val rightStruct = right.fullRowType

    val lRequested = requestedType.select(requestedType.fieldNames.filter(leftStruct.hasField))._1
    val rRequested = requestedType.select(requestedType.fieldNames.filter(rightStruct.hasField))._1

    (lRequested, rRequested)
  }

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val (lRequested, rRequested) = splitRequestedType(requestedType)
    val lRequired = left.rowRequiredness(lRequested)
    val rRequired = right.rowRequiredness(rRequested)

    RStruct.fromNamesAndTypes(requestedType.fieldNames.map(f => (f, lRequired.fieldType.getOrElse(f, rRequired.fieldType(f)))))
  }

  lazy val fullRowType: TStruct = {
    val leftStruct = left.fullRowType.deleteKey(left.uidFieldName)
    val rightStruct = right.fullRowType
    TStruct.concat(leftStruct, rightStruct)
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)

  override def emitStream(ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct
  ): IEmitCode = {
    val (lRequested, rRequested) = splitRequestedType(requestedType)

    context.toI(cb).flatMap(cb) { case zippedContext: SBaseStructValue =>
      val ctx1 = EmitCode.fromI(cb.emb)(zippedContext.loadField(_, "leftContext"))
      val ctx2 = EmitCode.fromI(cb.emb)(zippedContext.loadField(_, "rightContext"))
      left.emitStream(ctx, cb, mb, ctx1, lRequested).flatMap(cb) { sstream1 =>
        right.emitStream(ctx, cb, mb, ctx2, rRequested).map(cb) { sstream2 =>

          val stream1 = sstream1.asStream.getProducer(cb.emb)
          val stream2 = sstream2.asStream.getProducer(cb.emb)

          val region = cb.emb.genFieldThisRef[Region]("partition_zipped_reader_region")

          SStreamValue(new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb
            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              cb.assign(stream1.elementRegion, elementRegion)
              stream1.initialize(cb, outerRegion)
              cb.assign(stream2.elementRegion, elementRegion)
              stream2.initialize(cb, outerRegion)
            }

            override val elementRegion: Settable[Region] = region
            override val requiresMemoryManagementPerElement: Boolean =
              stream1.requiresMemoryManagementPerElement || stream2.requiresMemoryManagementPerElement

            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              cb.goto(stream1.LproduceElement)

              cb.define(stream1.LproduceElementDone)
              cb.goto(stream2.LproduceElement)

              cb.define(stream2.LproduceElementDone)
              cb.goto(LproduceElementDone)

              cb.define(stream1.LendOfStream)
              cb.goto(LendOfStream)

              cb.define(stream2.LendOfStream)
              cb._fatal("unexpected end of stream from right of zipped stream")
            }

            override val element: EmitCode = EmitCode.fromI(mb) { cb =>
              stream1.element.toI(cb).flatMap(cb) { case elt1: SBaseStructValue =>
                stream2.element.toI(cb).map(cb) { case elt2: SBaseStructValue =>
                  SBaseStruct.merge(cb, elt1.asBaseStruct, elt2.asBaseStruct)
                }
              }
            }

            override def close(cb: EmitCodeBuilder): Unit = {
              stream1.close(cb)
              stream2.close(cb)
            }
          })
        }
      }
    }
  }
}

case class PartitionZippedIndexedNativeReader(specLeft: AbstractTypedCodecSpec, specRight: AbstractTypedCodecSpec,
  indexSpecLeft: AbstractIndexSpec, indexSpecRight: AbstractIndexSpec,
  key: IndexedSeq[String], uidFieldName: String
) extends PartitionReader {

  def contextType: Type = {
    TStruct(
      "partitionIndex" -> TInt64,
      "leftPartitionPath" -> TString,
      "rightPartitionPath" -> TString,
      "indexPath" -> TString,
      "interval" -> RVDPartitioner.intervalIRRepresentation(specLeft.encodedVirtualType.asInstanceOf[TStruct].select(key)._1)
    )
  }

  private[this] def splitRequestedTypes(requestedType: Type): (TStruct, TStruct) = {
    val reqStruct = requestedType.asInstanceOf[TStruct]
    val neededFields = reqStruct.fieldNames.toSet

    val leftProvidedFields = specLeft.encodedVirtualType.asInstanceOf[TStruct].fieldNames.toSet
    val rightProvidedFields = specRight.encodedVirtualType.asInstanceOf[TStruct].fieldNames.toSet
    val leftNeededFields = leftProvidedFields.intersect(neededFields)
    val rightNeededFields = rightProvidedFields.intersect(neededFields)
    assert(leftNeededFields.intersect(rightNeededFields).isEmpty)

    val leftStruct = reqStruct.filterSet(leftNeededFields)._1
    val rightStruct = reqStruct.filterSet(rightNeededFields)._1
    (leftStruct, rightStruct)
  }

  def rowRequiredness(requestedType: TStruct): RStruct = {
    val (leftStruct, rightStruct) = splitRequestedTypes(requestedType)
    val rt = TypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    val pt = specLeft.decodedPType(leftStruct).asInstanceOf[PStruct].insertFields(specRight.decodedPType(rightStruct).asInstanceOf[PStruct].fields.map(f => (f.name, f.typ)))
    rt.fromPType(pt)
    rt
  }

  val uidSType: SStackStruct = SStackStruct(
    TTuple(TInt64, TInt64, TInt64, TInt64),
    Array(EmitType(SInt64, true), EmitType(SInt64, true), EmitType(SInt64, true), EmitType(SInt64, true)))

  def fullRowType: TStruct =
    (specLeft.encodedVirtualType.asInstanceOf[TStruct] ++ specRight.encodedVirtualType.asInstanceOf[TStruct])
      .insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64, TInt64, TInt64)))

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val (leftRType, rightRType) = splitRequestedTypes(requestedType)

    val insertUID: Boolean = requestedType.hasField(uidFieldName)

    val leftOffsetFieldIndex = indexSpecLeft.offsetFieldIndex
    val rightOffsetFieldIndex = indexSpecRight.offsetFieldIndex

    val index = new StagedIndexReader(cb.emb, indexSpecLeft.leafCodec, indexSpecLeft.internalNodeCodec)

    context.toI(cb).map(cb) { case _ctxStruct: SBaseStructValue =>
      val ctxStruct = cb.memoizeField(_ctxStruct, "ctxStruct").asBaseStruct

      val region = mb.genFieldThisRef[Region]("pnr_region")
      val partIdx = mb.genFieldThisRef[Long]("partIdx")
      val curIdx = mb.genFieldThisRef[Long]("curIdx")
      val endIdx = mb.genFieldThisRef[Long]("endIdx")

      val leftDec = specLeft.encodedType.buildDecoder(leftRType, mb.ecb)
      val rightDec = specRight.encodedType.buildDecoder(rightRType, mb.ecb)

      val leftBuffer = mb.genFieldThisRef[InputBuffer]("left_inputbuffer")
      val rightBuffer = mb.genFieldThisRef[InputBuffer]("right_inputbuffer")

      val leftValue = mb.newPField("leftValue", specLeft.encodedType.decodedSType(leftRType))
      val rightValue = mb.newPField("rightValue", specRight.encodedType.decodedSType(rightRType))

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = Some(_ => (endIdx - curIdx).toI)

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
          val indexPath = ctxStruct
            .loadField(cb, "indexPath")
            .get(cb)
            .asString
            .loadString(cb)
          val interval = ctxStruct
            .loadField(cb, "interval")
            .get(cb)
            .asInterval
          index.initialize(cb, indexPath)

          val indexResult = index.queryInterval(cb, interval)
          val startIndex = indexResult.loadField(cb, 0)
            .get(cb)
            .asInt64
            .value
          val endIndex = indexResult.loadField(cb, 1)
            .get(cb)
            .asInt64
            .value
          cb.assign(curIdx, startIndex)
          cb.assign(endIdx, endIndex)

          cb.assign(partIdx, ctxStruct.loadField(cb, "partitionIndex").get(cb).asInt64.value)
          cb.assign(leftBuffer, specLeft.buildCodeInputBuffer(
            Code.newInstance[ByteTrackingInputStream, InputStream](
              mb.openUnbuffered(ctxStruct.loadField(cb, "leftPartitionPath")
                .get(cb)
                .asString
                .loadString(cb), true))))
          cb.assign(rightBuffer, specRight.buildCodeInputBuffer(
            Code.newInstance[ByteTrackingInputStream, InputStream](
              mb.openUnbuffered(ctxStruct.loadField(cb, "rightPartitionPath")
                .get(cb)
                .asString
                .loadString(cb), true))))

          cb.ifx(endIndex > startIndex, {
            val leafNode = indexResult.loadField(cb, 2)
              .get(cb)
              .asBaseStruct

            val leftSeekAddr = leftOffsetFieldIndex match {
              case Some(offsetIdx) =>
                leafNode
                  .loadField(cb, "annotation")
                  .get(cb)
                  .asBaseStruct
                  .loadField(cb, offsetIdx)
                  .get(cb)
              case None =>
                leafNode
                  .loadField(cb, "offset")
                  .get(cb)
            }
            cb += leftBuffer.seek(leftSeekAddr.asInt64.value)

            val rightSeekAddr = rightOffsetFieldIndex match {
              case Some(offsetIdx) =>
                leafNode
                  .loadField(cb, "annotation")
                  .get(cb)
                  .asBaseStruct
                  .loadField(cb, offsetIdx)
                  .get(cb)
              case None =>
                leafNode
                  .loadField(cb, "offset")
                  .get(cb)
            }
            cb += rightBuffer.seek(rightSeekAddr.asInt64.value)
          })

          index.close(cb)
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(curIdx >= endIdx, cb.goto(LendOfStream))
          val nextLeft = leftBuffer.readByte()
          cb.ifx(nextLeft cne 1, cb._fatal(s"bad rows buffer state!"))
          val nextRight = rightBuffer.readByte()
          cb.ifx(nextRight cne 1, cb._fatal(s"bad entries buffer state!"))
          cb.assign(curIdx, curIdx + 1L)
          cb.assign(leftValue, leftDec(cb, region, leftBuffer))
          cb.assign(rightValue, rightDec(cb, region, rightBuffer))
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          if (insertUID) {
            val uid = SStackStruct.constructFromArgs(cb, region, TTuple(TInt64, TInt64),
              EmitCode.present(mb, primitive(partIdx)),
              EmitCode.present(mb, primitive(cb.memoize(curIdx.get - 1L))))
            val merged = SBaseStruct.merge(cb, leftValue.asBaseStruct, rightValue.asBaseStruct)
            IEmitCode.present(cb, merged._insert(requestedType, uidFieldName -> EmitValue.present(uid)))
          } else {
            IEmitCode.present(cb, SBaseStruct.merge(cb, leftValue.asBaseStruct, rightValue.asBaseStruct))
          }
        }

        override def close(cb: EmitCodeBuilder): Unit = {
          leftBuffer.invoke[Unit]("close")
          rightBuffer.invoke[Unit]("close")
        }
      }
      SStreamValue(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class TableNativeReaderParameters(
  path: String,
  options: Option[NativeReaderOptions])

class TableNativeReader(
  val params: TableNativeReaderParameters,
  val spec: AbstractTableSpec
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = Array(params.path)

  val filterIntervals: Boolean = params.options.map(_.filterIntervals).getOrElse(false)

  def partitionCounts: Option[IndexedSeq[Long]] = if (params.options.isDefined) None else Some(spec.partitionCounts)

  override def isDistinctlyKeyed: Boolean = spec.isDistinctlyKeyed

  def uidType = TTuple(TInt64, TInt64)

  def fullTypeWithoutUIDs = spec.table_type

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(tcoerce[PStruct](spec.rowsComponent.rvdSpec(ctx.fs, params.path)
      .typedCodecSpec.encodedType.decodedPType(requestedType.rowType)))

  protected def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(tcoerce[PStruct](spec.globalsComponent.rvdSpec(ctx.fs, params.path)
      .typedCodecSpec.encodedType.decodedPType(requestedType.globalType)))

  override def apply[M[_]: MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue] =
    lower(requestedType) >>= (TableStageIntermediate(_).asTableValue)

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "TableNativeReader")
  }

  override def renderShort(): String = s"(TableNativeReader ${ params.path } ${ params.options.map(_.renderShort()).getOrElse("") })"

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: TableNativeReader => params == that.params
    case _ => false
  }

  override def toString(): String = s"TableNativeReader(${ params })"

  override def lowerGlobals[M[_]](requestedGlobalsType: TStruct)(implicit M: MonadLower[M]): M[IR] =
    M.pure {
      val globalsSpec = spec.globalsSpec
      val globalsPath = spec.globalsComponent.absolutePath(params.path)
      assert(!requestedGlobalsType.hasField(uidFieldName))
      ArrayRef(
        ToArray(ReadPartition(
          MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> Str(globalsSpec.absolutePartPaths(globalsPath).head))),
          requestedGlobalsType,
          PartitionNativeReader(globalsSpec.typedCodecSpec, uidFieldName))),
        0)
    }

  override def lower[M[_]](requestedType: TableType)(implicit M: MonadLower[M]): M[TableStage] =
    for {
      globals <- lowerGlobals(requestedType.globalType)
      rowsSpec = spec.rowsSpec
      stateManager <- M.reader(_.stateManager)
      specPart = rowsSpec.partitioner(stateManager)

      partitioner =
        if (filterIntervals)
          params.options.map(opts => RVDPartitioner.union(stateManager, specPart.kType, opts.intervals, specPart.kType.size - 1))
        else
          params.options.map(opts => new RVDPartitioner(stateManager, specPart.kType, opts.intervals))

      // If the data on disk already has a uidFieldName field, we should read it
      // as is. Do this by passing a dummy uidFieldName to the rows component,
      // which is not in the requestedType, so is ignored.
      requestedUIDFieldName =
        if (spec.table_type.rowType.hasField(uidFieldName)) "__dummy_uid"
        else uidFieldName

      stage <- spec.rowsSpec.readTableStage(
        spec.rowsComponent.absolutePath(params.path),
        requestedType,
        requestedUIDFieldName,
        partitioner,
        filterIntervals
      )(globals)
    } yield stage
}

case class TableNativeZippedReader(
  pathLeft: String,
  pathRight: String,
  options: Option[NativeReaderOptions],
  specLeft: AbstractTableSpec,
  specRight: AbstractTableSpec
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = FastSeq(pathLeft, pathRight)

  override def renderShort(): String = s"(TableNativeZippedReader $pathLeft $pathRight ${ options.map(_.renderShort()).getOrElse("") })"

  private lazy val filterIntervals = options.exists(_.filterIntervals)

  private def intervals = options.map(_.intervals)

  require((specLeft.table_type.rowType.fieldNames ++ specRight.table_type.rowType.fieldNames).areDistinct())
  require(specRight.table_type.key.isEmpty)
  require(specLeft.partitionCounts sameElements specRight.partitionCounts)
  require(specLeft.version == specRight.version)

  def partitionCounts: Option[IndexedSeq[Long]] = if (intervals.isEmpty) Some(specLeft.partitionCounts) else None

  override def uidType = TTuple(TInt64, TInt64)

  override def fullTypeWithoutUIDs: TableType = specLeft.table_type.copy(
    rowType = specLeft.table_type.rowType ++ specRight.table_type.rowType)
  private val leftFieldSet = specLeft.table_type.rowType.fieldNames.toSet
  private val rightFieldSet = specRight.table_type.rowType.fieldNames.toSet

  def leftRType(requestedType: TStruct): TStruct =
    requestedType.filter(f => leftFieldSet.contains(f.name))._1

  def rightRType(requestedType: TStruct): TStruct =
    requestedType.filter(f => rightFieldSet.contains(f.name))._1

  def leftPType(ctx: ExecuteContext, leftRType: TStruct): PStruct =
    tcoerce[PStruct](specLeft.rowsComponent.rvdSpec(ctx.fs, pathLeft)
      .typedCodecSpec.encodedType.decodedPType(leftRType))

  def rightPType(ctx: ExecuteContext, rightRType: TStruct): PStruct =
    tcoerce[PStruct](specRight.rowsComponent.rvdSpec(ctx.fs, pathRight)
      .typedCodecSpec.encodedType.decodedPType(rightRType))

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(fieldInserter(ctx, leftPType(ctx, leftRType(requestedType.rowType)),
      rightPType(ctx, rightRType(requestedType.rowType)))._1)

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(specLeft.globalsComponent.rvdSpec(ctx.fs, pathLeft)
      .typedCodecSpec.encodedType.decodedPType(requestedType.globalType))

  def fieldInserter(ctx: ExecuteContext, pLeft: PStruct, pRight: PStruct)
  : (PStruct, (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3RegionLongLongLong) = {
    import Lower.monadLowerInstanceForLower
    val (Some(PTypeReferenceSingleCodeType(t: PStruct)), mk) =
      ir.Compile[Lower, AsmFunction3RegionLongLongLong](
        FastIndexedSeq("left" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pLeft)), "right" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pRight))),
        FastIndexedSeq(typeInfo[Region], LongInfo, LongInfo), LongInfo,
        InsertFields(Ref("left", pLeft.virtualType), pRight.fieldNames.map(f =>
            f -> GetField(Ref("right", pRight.virtualType), f))
        ))
        .runA(ctx, LoweringState())
    (t, mk)
  }

  override def apply[M[_]: MonadLower](requestedType: TableType, dropRows: Boolean): M[TableValue] =
    lower(requestedType) >>= (TableStageIntermediate(_).asTableValue)

  override def lowerGlobals[M[_]](requestedGlobalsType: TStruct)(implicit M: MonadLower[M]): M[IR] =
    M.pure {
      val globalsSpec = specLeft.globalsSpec
      val globalsPath = specLeft.globalsComponent.absolutePath(pathLeft)
      ArrayRef(
        ToArray(ReadPartition(
          MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> Str(globalsSpec.absolutePartPaths(globalsPath).head))),
          requestedGlobalsType,
          PartitionNativeReader(globalsSpec.typedCodecSpec, uidFieldName))),
        0
      )
    }

  override def lower[M[_]](requestedType: TableType)(implicit M: MonadLower[M]): M[TableStage] =
    for {
      globals <- lowerGlobals(requestedType.globalType)
      rowsSpec = specLeft.rowsSpec
      ctx <- M.ask
      specPart = rowsSpec.partitioner(ctx.stateManager)
      partitioner =
        if (filterIntervals) options.map(opts => RVDPartitioner.union(ctx.stateManager, specPart.kType, opts.intervals, specPart.kType.size - 1))
        else options.map(opts => new RVDPartitioner(ctx.stateManager, specPart.kType, opts.intervals))

      lowered <- AbstractRVDSpec.readZippedLowered(
        specLeft.rowsSpec, specRight.rowsSpec,
        pathLeft + "/rows", pathRight + "/rows",
        partitioner, filterIntervals,
        requestedType.rowType, requestedType.key, uidFieldName
      )(globals)
    } yield lowered

}

object TableFromBlockMatrixNativeReader {
  def apply(fs: FS, params: TableFromBlockMatrixNativeReaderParameters): TableFromBlockMatrixNativeReader = {
    val metadata: BlockMatrixMetadata = BlockMatrix.readMetadata(fs, params.path)
    TableFromBlockMatrixNativeReader(params, metadata)

  }

  def apply(fs: FS, path: String, nPartitions: Option[Int] = None, maximumCacheMemoryInBytes: Option[Int] = None): TableFromBlockMatrixNativeReader =
    TableFromBlockMatrixNativeReader(fs, TableFromBlockMatrixNativeReaderParameters(path, nPartitions, maximumCacheMemoryInBytes))

  def fromJValue(fs: FS, jv: JValue): TableFromBlockMatrixNativeReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[TableFromBlockMatrixNativeReaderParameters]
    TableFromBlockMatrixNativeReader(fs, params)
  }
}

case class TableFromBlockMatrixNativeReaderParameters(path: String, nPartitions: Option[Int], maximumCacheMemoryInBytes: Option[Int])

case class TableFromBlockMatrixNativeReader(
  params: TableFromBlockMatrixNativeReaderParameters,
  metadata: BlockMatrixMetadata
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = FastSeq(params.path)

  val getNumPartitions: Int = params.nPartitions.getOrElse(HailContext.backend.defaultParallelism)

  val partitionRanges = (0 until getNumPartitions).map { i =>
    val nRows = metadata.nRows
    val start = (i * nRows) / getNumPartitions
    val end = ((i + 1) * nRows) / getNumPartitions
    start until end
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = {
    Some(partitionRanges.map(r => r.end - r.start))
  }

  override def uidType = TInt64

  override def fullTypeWithoutUIDs: TableType = TableType(
    TStruct("row_idx" -> TInt64, "entries" -> TArray(TFloat64)),
    Array("row_idx"),
    TStruct.empty)

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.rowType).setRequired(true))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PInt64Required)

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct.empty(required = true))

  override def apply[M[_]](requestedType: TableType, dropRows: Boolean)
                          (implicit M: MonadLower[M]): M[TableValue] =
    M.ask.flatMap { ctx =>
      val rowsRDD = new BlockMatrixReadRowBlockedRDD(
        ctx.fsBc, params.path, partitionRanges, requestedType.rowType, metadata,
        maybeMaximumCacheMemoryInBytes = params.maximumCacheMemoryInBytes)

      val partitionBounds = partitionRanges.map { r => Interval(Row(r.start), Row(r.end), true, false) }
      val partitioner = new RVDPartitioner(ctx.stateManager, fullType.keyType, partitionBounds)

      val rowTyp = PType.canonical(requestedType.rowType, required = true).asInstanceOf[PStruct]
      val rvd = RVD(RVDType(rowTyp, fullType.key.filter(rowTyp.hasField)), partitioner, ContextRDD(rowsRDD))
      M.map(BroadcastRow.empty)(TableValue(requestedType, _, rvd))
    }

  override def toJValue: JValue = {
    decomposeWithName(params, "TableFromBlockMatrixNativeReader")(TableReader.formats)
  }

  def renderShort(): String = defaultRender()
}

object TableRead {
  def native(fs: FS, path: String, uidField: Boolean = false): TableRead = {
    val tr = TableNativeReader(fs, TableNativeReaderParameters(path, None))
    val requestedType = if (uidField)
     tr.fullType
    else
      tr.fullType.copy(
        rowType = tr.fullType.rowType.deleteKey(TableReader.uidFieldName))
    TableRead(requestedType, false, tr)
  }
}

case class TableRead(typ: TableType, dropRows: Boolean, tr: TableReader) extends TableIR {
  try {
    assert(PruneDeadFields.isSupertype(typ, tr.fullType))
  } catch {
    case e: Throwable =>
      fatal(s"bad type:\n  full type: ${tr.fullType}\n  requested: $typ\n  reader: $tr", e)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = if (dropRows) Some(FastIndexedSeq(0L)) else tr.partitionCounts

  def isDistinctlyKeyed: Boolean = tr.isDistinctlyKeyed

  lazy val rowCountUpperBound: Option[Long] = partitionCounts.map(_.sum)

  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(typ, dropRows, tr)
  }

  override protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    tr.apply(typ, dropRows).map(TableValueIntermediate).widen
}

case class TableParallelize(rowsAndGlobal: IR, nPartitions: Option[Int] = None) extends TableIR {
  require(rowsAndGlobal.typ.isInstanceOf[TStruct])
  require(rowsAndGlobal.typ.asInstanceOf[TStruct].fieldNames.sameElements(Array("rows", "global")))
  require(nPartitions.forall(_ > 0))

  lazy val rowCountUpperBound: Option[Long] = None

  private val rowsType = rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("rows").asInstanceOf[TArray]
  private val globalsType = rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global").asInstanceOf[TStruct]

  val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(rowsAndGlobal)

  def copy(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    val IndexedSeq(newrowsAndGlobal: IR) = newChildren
    TableParallelize(newrowsAndGlobal, nPartitions)
  }

  val typ: TableType = TableType(
    rowsType.elementType.asInstanceOf[TStruct],
    FastIndexedSeq(),
    globalsType)

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
   for {
    (ptype: PStruct, res) <-
      CompileAndEvaluate._apply(rowsAndGlobal, optimize = false).map {
        case Right((t, off)) => (t.fields(0).typ, t.loadField(off, 0))
      }

    globalsT = ptype.types(1).setRequired(true).asInstanceOf[PStruct]
    _ <- M.raiseWhen(ptype.isFieldMissing(res, 1)) {
      new HailException("'parallelize': found missing global value")
    }

    ctx <- M.ask
    globals = BroadcastRow(ctx.stateManager, RegionValue(ctx.r, ptype.loadField(res, 1)), globalsT)

    rowsT = ptype.types(0).asInstanceOf[PArray]
    rowT = rowsT.elementType.asInstanceOf[PStruct].setRequired(true)
    spec = TypedCodecSpec(rowT, BufferSpec.wireSpec)

    makeEnc = spec.buildEncoder(ctx, rowT)
    rowsAddr = ptype.loadField(res, 0)
    nRows = rowsT.loadLength(rowsAddr)

    nSplits = math.min(nPartitions.getOrElse(16), math.max(nRows, 1))
    parts = partition(nRows, nSplits)

    bae = new ByteArrayEncoder(ctx.theHailClassLoader, makeEnc)
    encRows = {
      var idx = 0
      Array.tabulate(nSplits) { splitIdx =>
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
    }

    (resultRowType: PStruct, makeDec) = spec.buildDecoder(ctx, typ.rowType)
    _ <- assertA(resultRowType.virtualType == typ.rowType,
      s"typ mismatch:\n  res=${ resultRowType.virtualType }\n  typ=${ typ.rowType }"
    )

    _ = log.info(s"parallelized $nRows rows in $nSplits partitions")

    rvd = ContextRDD.parallelize(encRows, encRows.length)
      .cmapPartitions { (ctx, it) =>
        it.flatMap { case (nRowPartition, arr) =>
          val bais = new ByteArrayDecoder(theHailClassLoaderForSparkWorkers, makeDec)
          bais.set(arr)
          Iterator.range(0, nRowPartition)
            .map { _ =>
              bais.readValue(ctx.region)
            }
        }
      }

  } yield TableValueIntermediate(TableValue(typ, globals, RVD.unkeyed(resultRowType, rvd)))
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = keys)

  def definitelyDoesNotShuffle: Boolean = child.typ.key.startsWith(keys) || isSorted

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, isSorted)
  }

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {tv <- child.execute(r) >>= (_.asTableValue); ctx <- M.ask}
      yield TableValueIntermediate(tv.copy(typ = typ, rvd = tv.rvd.enforceKey(ctx, keys, isSorted)))
}

/**
 * Generate a table from the elementwise application of a body IR to a stream of `contexts`.
 *
 * @param contexts IR of type TStream[Any] whose elements are downwardly exposed to `body` as `cname`.
 * @param globals  IR of type TStruct, downwardly exposed to `body` as `gname`.
 * @param cname    Name of free variable in `body` referencing elements of `contexts`.
 * @param gname    Name of free variable in `body` referencing `globals`.
 * @param body     IR of type TStream[TStruct] that generates the rows of the table for each
 *                 element in `contexts`, optionally referencing free variables Ref(cname) and
 *                 Ref(gname).
 * @param partitioner
 * @param errorId  Identifier tracing location in Python source that created this node
 */
case class TableGen(contexts: IR,
                    globals: IR,
                    cname: String,
                    gname: String,
                    body: IR,
                    partitioner: RVDPartitioner,
                    errorId: Int = ErrorIDs.NO_ERROR
                   ) extends TableIR {

  TypeCheck.coerce[TStream]("contexts", contexts.typ)

  private val globalType =
    TypeCheck.coerce[TStruct]("globals", globals.typ)

  private val rowType = {
    val bodyType = TypeCheck.coerce[TStream]( "body", body.typ)
    TypeCheck.coerce[TStruct]( "body.elementType", bodyType.elementType)
  }

  if (!partitioner.kType.isSubsetOf(rowType))
    throw new IllegalArgumentException(
      s"""'partitioner': key type contains fields absent from row type
         |  Key type: ${partitioner.kType}
         |  Row type: $rowType""".stripMargin
    )

  override def typ: TableType =
    TableType(rowType, partitioner.kType.fieldNames, globalType)

  override val rowCountUpperBound: Option[Long] =
    None

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(contexts: IR, globals: IR, body: IR) = newChildren
    TableGen(contexts, globals, cname, gname, body, partitioner, errorId)
  }

  override def childrenSeq: IndexedSeq[BaseIR] =
    FastSeq(contexts, globals, body)

  override protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    for {
      analyses <- LoweringAnalyses(this)
      lowered <- LowerTableIR.applyTable(this, DArrayLowering.All, analyses)
    } yield TableStageIntermediate(lowered)
}

case class TableRange(n: Int, nPartitions: Int) extends TableIR {
  require(n >= 0)
  require(nPartitions > 0)
  private val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRange = {
    assert(newChildren.isEmpty)
    TableRange(n, nPartitions)
  }

  private val partCounts = partition(n, nPartitionsAdj)

  override val partitionCounts = Some(partCounts.map(_.toLong).toFastIndexedSeq)

  lazy val rowCountUpperBound: Option[Long] = Some(n.toLong)

  val typ: TableType = TableType(
    TStruct("idx" -> TInt32),
    Array("idx"),
    TStruct.empty)

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    M.map2(M.ask, BroadcastRow.empty) { (ctx, br) =>
      val localRowType = PCanonicalStruct(true, "idx" -> PInt32Required)
      val localPartCounts = partCounts
      val partStarts = partCounts.scanLeft(0)(_ + _)
      TableValueIntermediate(TableValue(typ,
        br,
        new RVD(
          RVDType(localRowType, Array("idx")),
          new RVDPartitioner(ctx.stateManager, Array("idx"), typ.rowType,
            Array.tabulate(nPartitionsAdj) { i =>
              val start = partStarts(i)
              val end = partStarts(i + 1)
              Interval(Row(start), Row(end), includesStart = true, includesEnd = false)
            }),
          ContextRDD.parallelize(Range(0, nPartitionsAdj), nPartitionsAdj)
            .cmapPartitionsWithIndex { case (i, ctx, _) =>
              val region = ctx.region

              val start = partStarts(i)
              Iterator.range(start, start + localPartCounts(i))
                .map { j =>
                  val off = localRowType.allocate(region)
                  localRowType.setFieldPresent(off, 0)
                  Region.storeInt(localRowType.fieldOffset(off, 0), j)
                  off
                }
            }
        )
      ))
    }
}

case class TableFilter(child: TableIR, pred: IR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, pred)

  val typ: TableType = child.typ

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableFilter = {
    assert(newChildren.length == 2)
    TableFilter(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M])
  : M[TableExecuteIntermediate] = {
    val readTV = child.execute(r) >>= (_.asTableValue)
    pred match {
      case True() =>
        readTV.map(TableValueIntermediate)

      case False() =>
        M.map2(readTV, M.ask) { case (tv, ctx) =>
          TableValueIntermediate(tv.copy(rvd = RVD.empty(ctx, typ.canonicalRVDType)))
        }

      case _ =>
        for {
          tv <- readTV
          (Some(BooleanSingleCodeType), f) <-
            ir.Compile[M, AsmFunction3RegionLongLongBoolean](
              FastIndexedSeq(
                "row" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)),
                "global" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))
              ),
              FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), BooleanInfo,
              Coalesce(FastIndexedSeq(pred, False()))
            )

          filtered <- tv.filterWithPartitionOp(f) {
            case (rowF, ctx, ptr, globalPtr) => rowF(ctx.region, ptr, globalPtr)
          }
        } yield TableValueIntermediate(filtered)
    }
  }
}

object TableSubset {
  val HEAD: Int = 0
  val TAIL: Int = 1
}

trait TableSubset extends TableIR {
  val subsetKind: Int
  val child: TableIR
  val n: Long

  def typ: TableType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    child.partitionCounts.map(subsetKind match {
      case TableSubset.HEAD => PartitionCounts.getHeadPCs(_, n)
      case TableSubset.TAIL => PartitionCounts.getTailPCs(_, n)
    })

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound match {
    case Some(c) => Some(c.min(n))
    case None => Some(n)
  }

  override protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    for {prev <- child.execute(r) >>= (_.asTableValue)}
      yield TableValueIntermediate(prev.copy(rvd = subsetKind match {
        case TableSubset.HEAD => prev.rvd.head(n, child.partitionCounts)
        case TableSubset.TAIL => prev.rvd.tail(n, child.partitionCounts)
      }))
}

case class TableHead(child: TableIR, n: Long) extends TableSubset {
  require(n >= 0, fatal(s"TableHead: n must be non-negative! Found '$n'."))
  val subsetKind = TableSubset.HEAD

  def copy(newChildren: IndexedSeq[BaseIR]): TableHead = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableHead(newChild, n)
  }
}

case class TableTail(child: TableIR, n: Long) extends TableSubset {
  require(n >= 0, fatal(s"TableTail: n must be non-negative! Found '$n'."))
  val subsetKind = TableSubset.TAIL

  def copy(newChildren: IndexedSeq[BaseIR]): TableTail = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableTail(newChild, n)
  }
}

object RepartitionStrategy {
  val SHUFFLE: Int = 0
  val COALESCE: Int = 1
  val NAIVE_COALESCE: Int = 2
}

case class TableRepartition(child: TableIR, n: Int, strategy: Int) extends TableIR {
  def typ: TableType = child.typ

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableRepartition = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRepartition(newChild, n, strategy)
  }

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {prev <- child.execute(r) >>= (_.asTableValue); ctx <- M.ask}
      yield TableValueIntermediate(
        prev.copy(rvd = strategy match {
          case RepartitionStrategy.SHUFFLE => prev.rvd.coalesce(ctx, n, shuffle = true)
          case RepartitionStrategy.COALESCE => prev.rvd.coalesce(ctx, n, shuffle = false)
          case RepartitionStrategy.NAIVE_COALESCE => prev.rvd.naiveCoalesce(n, ctx)
        })
      )
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
    joinType == "outer")

  val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  lazy val rowCountUpperBound: Option[Long] = None

  private val newRowType = {
    val leftRowType = left.typ.rowType
    val rightRowType = right.typ.rowType
    val leftKey = left.typ.key.take(joinKey)
    val rightKey = right.typ.key.take(joinKey)

    val leftKeyType = TableType.keyType(leftRowType, leftKey)
    val leftValueType = TableType.valueType(leftRowType, leftKey)
    val rightValueType = TableType.valueType(rightRowType, rightKey)
    if (leftValueType.fieldNames.toSet
      .intersect(rightValueType.fieldNames.toSet)
      .nonEmpty)
      throw new RuntimeException(s"invalid join: \n  left value:  $leftValueType\n  right value: $rightValueType")

    leftKeyType ++ leftValueType ++ rightValueType
  }

  private val newGlobalType = left.typ.globalType ++ right.typ.globalType

  private val newKey = left.typ.key ++ right.typ.key.drop(joinKey)

  val typ: TableType = TableType(newRowType, newKey, newGlobalType)

  def copy(newChildren: IndexedSeq[BaseIR]): TableJoin = {
    assert(newChildren.length == 2)
    TableJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      joinType,
      joinKey)
  }

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      leftTV <- left.execute(r) >>= (_.asTableStage)
      rightTV <- right.execute(r) >>= (_.asTableStage)
      joined <- LowerTableIRHelpers.lowerTableJoin(r, this, leftTV, rightTV)
    } yield TableStageIntermediate(joined)
}

case class TableIntervalJoin(
  left: TableIR,
  right: TableIR,
  root: String,
  product: Boolean
) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  lazy val rowCountUpperBound: Option[Long] = left.rowCountUpperBound

  val rightType: Type = if (product) TArray(right.typ.valueType) else right.typ.valueType
  val typ: TableType = left.typ.copy(rowType = left.typ.rowType.appendKey(root, rightType))

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR =
    TableIntervalJoin(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[TableIR], root, product)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      leftValue <- left.execute(r) >>= (_.asTableValue)
      rightValue <- right.execute(r) >>= (_.asTableValue)

      leftRVDType = leftValue.rvd.typ
      rightRVDType = rightValue.rvd.typ.copy(key = rightValue.typ.key)
      rightValueFields = rightRVDType.valueType.fieldNames

      ctx <- M.ask
      sm = ctx.stateManager
      localKey = typ.key
      localRoot = root
      newRVD =
        if (product) {
          val joiner = (rightPType: PStruct) => {
            val leftRowType = leftRVDType.rowType
            val newRowType = leftRowType.appendKey(localRoot, PCanonicalArray(rightPType.selectFields(rightValueFields)))
            (RVDType(newRowType, localKey), (_: RVDContext, it: Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => {
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
            })
          }

          leftValue.rvd.orderedLeftIntervalJoin(ctx, rightValue.rvd, joiner)
        } else {
          val joiner = (rightPType: PStruct) => {
            val leftRowType = leftRVDType.rowType
            val newRowType = leftRowType.appendKey(localRoot, rightPType.selectFields(rightValueFields).setRequired(false))

            (RVDType(newRowType, localKey), (_: RVDContext, it: Iterator[JoinedRegionValue]) => {
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
            })
          }

          leftValue.rvd.orderedLeftIntervalJoinDistinct(ctx, rightValue.rvd, joiner)
        }

    } yield TableValueIntermediate(TableValue(typ, leftValue.globals, newRVD))
}

/**
  * The TableMultiWayZipJoin node assumes that input tables have distinct keys. If inputs
  * do not have distinct keys, the key that is included in the result is undefined, but
  * is likely the last.
  */
case class TableMultiWayZipJoin(childrenSeq: IndexedSeq[TableIR], fieldName: String, globalName: String) extends TableIR {
  require(childrenSeq.length > 0, "there must be at least one table as an argument")

  private val first = childrenSeq.head
  private val rest = childrenSeq.tail

  lazy val rowCountUpperBound: Option[Long] = None

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

  def copy(newChildren: IndexedSeq[BaseIR]): TableMultiWayZipJoin =
    TableMultiWayZipJoin(newChildren.asInstanceOf[IndexedSeq[TableIR]], fieldName, globalName)

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      childValues <- childrenSeq.traverse(_.execute(r).flatMap(_.asTableValue))
      ctx <- M.ask
      childRVDs = RVD.unify(ctx, childValues.map(_.rvd)).toFastIndexedSeq
      _ <- assertA(childRVDs.forall(_.typ.key.startsWith(typ.key)))

      repartitionedRVDs =
        if (childRVDs(0).partitioner.satisfiesAllowedOverlap(typ.key.length - 1) &&
          childRVDs.forall(rvd => rvd.partitioner == childRVDs(0).partitioner))
          childRVDs.map(_.truncateKey(typ.key.length))
        else {
          info("TableMultiWayZipJoin: repartitioning children")
          val childRanges = childRVDs.flatMap(_.partitioner.coarsenedRangeBounds(typ.key.length))
          val newPartitioner = RVDPartitioner.generate(ctx.stateManager, typ.keyType, childRanges)
          childRVDs.map(_.repartition(ctx, newPartitioner))
        }

      newPartitioner = repartitionedRVDs(0).partitioner

      rvdType = repartitionedRVDs(0).typ
      keyFields = rvdType.kType.fields.map(f => (f.name, f.typ))
      valueFields = rvdType.valueType.fields.map(f => (f.name, f.typ))
      localNewRowType = PCanonicalStruct(required = true,
        keyFields ++ Array((fieldName, PCanonicalArray(
          PCanonicalStruct(required = false, valueFields: _*), required = true))): _*)
      localDataLength = childrenSeq.length
      rvMerger = { (rvdCtx: RVDContext, it: Iterator[BoxedArrayBuilder[(RegionValue, Int)]]) =>
        val rvb = new RegionValueBuilder(ctx.stateManager)
        val newRegionValue = RegionValue()

        it.map { rvs =>
          val rv = rvs(0)._1
          rvb.set(rvdCtx.region)
          rvb.start(localNewRowType)
          rvb.startStruct()
          rvb.addFields(rvdType.rowType, rv, rvdType.kFieldIdx) // Add the key
          rvb.startMissingArray(localDataLength) // add the values
          var i = 0
          while (i < rvs.length) {
            val (rv, j) = rvs(i)
            rvb.setArrayIndex(j)
            rvb.setPresent()
            rvb.startStruct()
            rvb.addFields(rvdType.rowType, rv, rvdType.valueFieldIdx)
            rvb.endStruct()
            i += 1
          }
          rvb.endArrayUnchecked()
          rvb.endStruct()

          newRegionValue.set(rvb.region, rvb.end())
          newRegionValue
        }
      }

      rvd = RVD(
        typ = RVDType(localNewRowType, typ.key),
        partitioner = newPartitioner,
        crdd = ContextRDD.czipNPartitions(repartitionedRVDs.map(_.crdd.toCRDDRegionValue)) { (rvdCtx, its) =>
          val orvIters = its.map(it => OrderedRVIterator(rvdType, it, rvdCtx, ctx.stateManager))
          rvMerger(rvdCtx, OrderedRVIterator.multiZipJoin(ctx.stateManager, orvIters))
        }.toCRDDPtr
      )

      newGlobals <- BroadcastRow(Row(childValues.map(_.globals.javaValue)), newGlobalType)

    } yield TableValueIntermediate(TableValue(typ, newGlobals, rvd))
}

case class TableLeftJoinRightDistinct(left: TableIR, right: TableIR, root: String) extends TableIR {
  require(right.typ.keyType isPrefixOf left.typ.keyType,
    s"\n  L: ${ left.typ }\n  R: ${ right.typ }")

  lazy val rowCountUpperBound: Option[Long] = left.rowCountUpperBound

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  private val newRowType = left.typ.rowType.structInsert(right.typ.valueType, List(root))._1
  val typ: TableType = left.typ.copy(rowType = newRowType)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  def copy(newChildren: IndexedSeq[BaseIR]): TableLeftJoinRightDistinct = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableLeftJoinRightDistinct(newLeft, newRight, root)
  }

  override protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    for {
      leftValue <- left.execute(r) >>= (_.asTableValue)
      rightValue <- right.execute(r) >>= (_.asTableValue)
      joinKey = math.min(left.typ.key.length, right.typ.key.length)
    } yield TableValueIntermediate(leftValue.copy(
      typ = typ,
      rvd = leftValue.rvd
        .orderedLeftJoinDistinctAndInsert(rightValue.rvd.truncateKey(joinKey), root)
    ))
}

object TableMapPartitions {
  def apply(child: TableIR,
    globalName: String,
    partitionStreamName: String,
    body: IR): TableMapPartitions = TableMapPartitions(child, globalName, partitionStreamName, body, 0, child.typ.key.length)
}
case class TableMapPartitions(child: TableIR,
  globalName: String,
  partitionStreamName: String,
  body: IR,
  requestedKey: Int,
  allowedOverlap: Int
) extends TableIR {
  assert(body.typ.isInstanceOf[TStream], s"${ body.typ }")
  assert(allowedOverlap >= -1 && allowedOverlap <= child.typ.key.size)
  assert(requestedKey >= 0 && requestedKey <= child.typ.key.size)

  lazy val typ = child.typ.copy(
    rowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct])

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, body)

  val rowCountUpperBound: Option[Long] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): TableMapPartitions = {
    assert(newChildren.length == 2)
    TableMapPartitions(newChildren(0).asInstanceOf[TableIR],
      globalName, partitionStreamName, newChildren(1).asInstanceOf[IR], requestedKey, allowedOverlap)
  }

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      tv <- child.execute(r) >>= (_.asTableValue)
      rowPType = tv.rvd.rowPType
      globalPType = tv.globals.t

      (newRowPType: PStruct, makeIterator) <-
        CompileIterator.forTableMapPartitions(
          globalPType, rowPType,
          Subst(body, BindingEnv(Env(
            globalName -> In(0, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globalPType))),
            partitionStreamName -> In(1, SingleCodeEmitParamType(true, StreamSingleCodeType(requiresMemoryManagementPerElement = true, rowPType, true))))))
        )

      globalsBc <- tv.globals.broadcast
      ctx <- M.ask
      fsBc = ctx.fsBc
      itF = { (idx: Int, consumerCtx: RVDContext, partition: RVDContext => Iterator[Long]) =>
        val boxedPartition = new NoBoxLongIterator {
          var eos: Boolean = false
          var iter: Iterator[Long] = _

          override def init(partitionRegion: Region, elementRegion: Region): Unit = {
            iter = partition(new RVDContext(partitionRegion, elementRegion))
          }

          override def next(): Long = {
            if (!iter.hasNext) {
              eos = true
              0L
            } else
              iter.next()
          }

          override def close(): Unit = ()
        }

        makeIterator(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), consumerCtx,
          globalsBc.value.readRegionValue(consumerCtx.partitionRegion, theHailClassLoaderForSparkWorkers),
          boxedPartition
        ).map(_.longValue())
      }

      rvd = tv.rvd.repartition(ctx, tv.rvd.partitioner.strictify(allowedOverlap))

    } yield TableValueIntermediate(tv.copy(typ = typ,
      rvd = rvd.mapPartitionsWithContextAndIndex(RVDType(newRowPType, typ.key))(itF)
    ))
}

// Must leave key fields unchanged.
case class TableMapRows(child: TableIR, newRow: IR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, newRow)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val typ: TableType = child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      tv <- child.execute(r) >>= (_.asTableValue)
      scanRef = genUID()
      extracted = agg.Extract.apply(newRow, scanRef, r.requirednessAnalysis, isScan = true)
      intermediate <- if (extracted.aggs.isEmpty) {
        for {
          (Some(PTypeReferenceSingleCodeType(rTyp)), f) <-
            ir.Compile[M, AsmFunction3RegionLongLongLong](
              FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
                ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))
              ),
              FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
              Coalesce(FastIndexedSeq(
                extracted.postAggIR,
                Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ)
              ))
            )

          rowIterationNeedsGlobals = Mentions(extracted.postAggIR, "global")
          globalsBc <- if (rowIterationNeedsGlobals) tv.globals.broadcast else M.pure(null)
          fsBc <- M.reader(_.fsBc)
        } yield TableValueIntermediate(tv.copy(typ = typ,
          rvd = tv.rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key)) {
            (i: Int, rvdCtx: RVDContext, it: Iterator[Long]) =>
              val globalRegion = rvdCtx.partitionRegion
              val globals =
                if (rowIterationNeedsGlobals)
                  globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
                else
                  0

              val newRow = f(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
              it.map(r => newRow(rvdCtx.r, globals, r))
          }
        ))
      } else {
        val scanInitNeedsGlobals = Mentions(extracted.init, "global")
        val scanSeqNeedsGlobals = Mentions(extracted.seqPerElt, "global")
        val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, "global")

        for {
          globalsBc <-
            if (rowIterationNeedsGlobals || scanInitNeedsGlobals || scanSeqNeedsGlobals)
             tv.globals.broadcast
            else
              M.pure(null)

          spec = BufferSpec.blockedUncompressed

          // Order of operations:
          // 1. init op on all aggs and serialize to byte array.
          // 2. load in init op on each partition, seq op over partition, serialize.
          // 3. load in partition aggregations, comb op as necessary, serialize.
          // 4. load in partStarts, calculate newRow based on those results.

          (_, initF) <- ir.CompileWithAggregators[M, AsmFunction2RegionLongUnit](
            extracted.states,
            FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t)))),
            FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
            Begin(FastIndexedSeq(extracted.init))
          )

          serializeF <- extracted.serialize(spec)

          ctx <- M.ask
          (_, eltSeqF) <- ir.CompileWithAggregators[M, AsmFunction3RegionLongLongUnit](
            extracted.states,
            FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
              ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))),
            FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
            extracted.eltOp(ctx)
          )

          read <- extracted.deserialize(spec)
          write <- extracted.serialize(spec)
          combOpFNeedsPool <- extracted.combOpFSerializedFromRegionPool(spec)

          (Some(PTypeReferenceSingleCodeType(rTyp)), f) <-
            ir.CompileWithAggregators[M, AsmFunction3RegionLongLongLong](
              extracted.states,
              FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
                ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))),
              FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
              Let(scanRef, extracted.results, Coalesce(FastIndexedSeq(
                extracted.postAggIR,
                Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ))
              ))
            )

          _ <- assertA(rTyp.virtualType == newRow.typ)
          fsBc <- M.reader(_.fsBc)

          // 1. init op on all aggs and write out to initPath
          initAgg = ctx.r.pool.scopedRegion { aggRegion =>
            ctx.r.pool.scopedRegion { fRegion =>
              val init = initF(ctx.theHailClassLoader, fsBc.value, ctx.taskContext, fRegion)
              init.newAggState(aggRegion)
              init(fRegion, tv.globals.value.offset)
              serializeF(ctx.theHailClassLoader, ctx.taskContext, aggRegion, init.getAggOffset())
            }
          }
        } yield
          if (ctx.getFlag("distributed_scan_comb_op") != null && extracted.shouldTreeAggregate) {
            val tmpBase = ctx.createTmpPath("table-map-rows-distributed-scan")
            val d = digitsNeeded(tv.rvd.getNumPartitions)
            val files = tv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
              val path = tmpBase + "/" + partFile(d, i, TaskContext.get)
              val globalRegion = ctx.freshRegion()
              val globals = if (scanSeqNeedsGlobals) globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers) else 0

              ctx.r.pool.scopedSmallRegion { aggRegion =>
                val tc = SparkTaskContext.get()
                val seq = eltSeqF(theHailClassLoaderForSparkWorkers, fsBc.value, tc, globalRegion)

                seq.setAggState(aggRegion, read(theHailClassLoaderForSparkWorkers, tc, aggRegion, initAgg))
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

              filesToMerge = ContextRDD.weaken(SparkBackend.sparkContext("TableMapRows.execute").parallelize(0 until nToMerge, nToMerge))
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
                    val bytes = combOpFNeedsPool(() => (ctx.r.pool, theHailClassLoaderForSparkWorkers, SparkTaskContext.get()))(b1, b2)
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

                  b = combOpFNeedsPool(() => (ctx.r.pool, theHailClassLoaderForSparkWorkers, SparkTaskContext.get()))(b, using(new DataInputStream(fsBc.value.open(path)))(readToBytes))
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

            TableValueIntermediate(
              tv.copy(
                typ = typ,
                rvd = tv.rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key))(itF)
              )
            )
          } else {

            // 2. load in init op on each partition, seq op over partition, write out.
            val scanPartitionAggs = SpillingCollectIterator(ctx.localTmpdir, ctx.fs, tv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
              val globalRegion = ctx.partitionRegion
              val globals = if (scanSeqNeedsGlobals) globalsBc.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers) else 0

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
            }, ctx.getFlag("max_leader_scans").toInt)

            // 3. load in partition aggregations, comb op as necessary, write back out.
            val partAggs = scanPartitionAggs.scanLeft(initAgg)(combOpFNeedsPool(() => (ctx.r.pool, ctx.theHailClassLoader, ctx.taskContext)))
            val scanAggCount = tv.rvd.getNumPartitions
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
              val globals =
                if (rowIterationNeedsGlobals || scanSeqNeedsGlobals)
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

            TableValueIntermediate(
              tv.copy(
                typ = typ,
                rvd = tv.rvd.mapPartitionsWithIndexAndValue(RVDType(rTyp.asInstanceOf[PStruct], typ.key), partitionIndices)(itF)
              )
            )
          }
      }
    } yield intermediate
}

case class TableMapGlobals(child: TableIR, newGlobals: IR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, newGlobals)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val typ: TableType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      tv <- child.execute(r) >>= (_.asTableValue)
      (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) <-
        Compile[M, AsmFunction2RegionLongLong](
          FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t)))),
          FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
          Coalesce(FastIndexedSeq(
            newGlobals,
            Die("Internal error: TableMapGlobals: globals missing", newGlobals.typ))
          )
        )

      ctx <- M.ask
      resultOff = f(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)(ctx.r, tv.globals.value.offset)
    } yield TableValueIntermediate(tv.copy(typ = typ,
      globals = BroadcastRow(ctx.stateManager, RegionValue(ctx.r, resultOff), resultPType)
    ))
}

case class TableExplode(child: TableIR, path: IndexedSeq[String]) extends TableIR {
  assert(path.nonEmpty)
  assert(!child.typ.key.contains(path.head))

  lazy val rowCountUpperBound: Option[Long] = None

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  private val childRowType = child.typ.rowType

  private val length: IR = {
    Coalesce(FastIndexedSeq(
      ArrayLen(CastToArray(
        path.foldLeft[IR](Ref("row", childRowType))((struct, field) =>
          GetField(struct, field)))),
      0))
  }

  val idx = Ref(genUID(), TInt32)
  val newRow: InsertFields = {
    val refs = path.init.scanLeft(Ref("row", childRowType))((struct, name) =>
      Ref(genUID(), tcoerce[TStruct](struct.typ).field(name).typ))

    path.zip(refs).zipWithIndex.foldRight[IR](idx) {
      case (((field, ref), i), arg) =>
        InsertFields(ref, FastIndexedSeq(field ->
          (if (i == refs.length - 1)
            ArrayRef(CastToArray(GetField(ref, field)), arg)
          else
            Let(refs(i + 1).name, GetField(ref, field), arg))))
    }.asInstanceOf[InsertFields]
  }

  val typ: TableType = child.typ.copy(rowType = newRow.typ)

  def copy(newChildren: IndexedSeq[BaseIR]): TableExplode = {
    assert(newChildren.length == 1)
    TableExplode(newChildren(0).asInstanceOf[TableIR], path)
  }

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      prev <- child.execute(r) >>= (_.asTableValue)

      (_, l) <- Compile[M, AsmFunction2RegionLongInt](
        FastIndexedSeq("row" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType))),
        FastIndexedSeq(classInfo[Region], LongInfo),
        IntInfo,
        length
      )

      (Some(PTypeReferenceSingleCodeType(newRowType: PStruct)), f) <-
        Compile[M, AsmFunction3RegionLongIntLong](
          FastIndexedSeq(
            "row" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType)),
            idx.name -> SingleCodeEmitParamType(true, Int32SingleCodeType)
          ),
          FastIndexedSeq(classInfo[Region], LongInfo, IntInfo),
          LongInfo,
          newRow
        )

      _ <- assertA(newRowType.virtualType == typ.rowType)

      rvdType: RVDType = RVDType(newRowType,
        prev.rvd.typ.key.takeWhile(_ != path.head)
      )

      fsBc <- M.reader(_.fsBc)
    } yield TableValueIntermediate(
      TableValue(typ,
        prev.globals,
        prev.rvd.boundary.mapPartitionsWithIndex(rvdType) { (i, ctx, it) =>
          val globalRegion = ctx.partitionRegion
          val lenF = l(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
          val rowF = f(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), globalRegion)
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
        })
    )
}

case class TableUnion(childrenSeq: IndexedSeq[TableIR]) extends TableIR {
  assert(childrenSeq.nonEmpty)
  assert(childrenSeq.tail.forall(_.typ.rowType == childrenSeq(0).typ.rowType))
  assert(childrenSeq.tail.forall(_.typ.key == childrenSeq(0).typ.key))

  lazy val rowCountUpperBound: Option[Long] = {
    val definedChildren = childrenSeq.flatMap(_.rowCountUpperBound)
    if (definedChildren.length == childrenSeq.length)
      Some(definedChildren.sum)
    else
      None
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableUnion = {
    TableUnion(newChildren.map(_.asInstanceOf[TableIR]))
  }

  val typ: TableType = childrenSeq(0).typ

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {tvs <- childrenSeq.traverse(_.execute(r) >>= (_.asTableValue)); ctx <- M.ask}
      yield TableValueIntermediate(tvs(0).copy(
        rvd = RVD.union(RVD.unify(ctx, tvs.map(_.rvd)), tvs(0).typ.key.length, ctx)
      ))
}

case class MatrixRowsTable(child: MatrixIR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsTable = {
    assert(newChildren.length == 1)
    MatrixRowsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.rowsTableType
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.colsTableType
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.entriesTableType
}

case class TableDistinct(child: TableIR) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableDistinct = {
    val IndexedSeq(newChild) = newChildren
    TableDistinct(newChild.asInstanceOf[TableIR])
  }

  val typ: TableType = child.typ


  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {prev <- child.execute(r) >>= (_.asTableValue); ctx <- M.ask}
      yield TableValueIntermediate(prev.copy(rvd = prev.rvd.truncateKey(prev.typ.key).distinctByKey(ctx)))
}

case class TableKeyByAndAggregate(
  child: TableIR,
  expr: IR,
  newKey: IR,
  nPartitions: Option[Int] = None,
  bufferSize: Int) extends TableIR {
  require(expr.typ.isInstanceOf[TStruct])
  require(newKey.typ.isInstanceOf[TStruct])
  require(bufferSize > 0)

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, expr, newKey)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyByAndAggregate = {
    val IndexedSeq(newChild: TableIR, newExpr: IR, newNewKey: IR) = newChildren
    TableKeyByAndAggregate(newChild, newExpr, newNewKey, nPartitions, bufferSize)
  }

  private val keyType = newKey.typ.asInstanceOf[TStruct]
  val typ: TableType = TableType(rowType = keyType ++ tcoerce[TStruct](expr.typ),
    globalType = child.typ.globalType,
    key = keyType.fieldNames
  )

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      prev <- child.execute(r) >>= (_.asTableValue)
      ctx <- M.ask
      fsBc = ctx.fsBc
      sm = ctx.stateManager

      localKeyType = keyType
      (Some(PTypeReferenceSingleCodeType(localKeyPType: PStruct)), makeKeyF) <-
        ir.Compile[M, AsmFunction3RegionLongLongLong](
          FastIndexedSeq(("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType))),
            ("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
          FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
          Coalesce(FastIndexedSeq(
            newKey,
            Die("Internal error: TableKeyByAndAggregate: newKey missing", newKey.typ)))
        )

      globalsBc <- prev.globals.broadcast
      res = genUID()
      extracted = agg.Extract(expr, res, r.requirednessAnalysis)

      (_, makeInit) <- ir.CompileWithAggregators[M, AsmFunction2RegionLongUnit](
        extracted.states,
        FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
        FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
        extracted.init
      )

      (_, makeSeq) <- ir.CompileWithAggregators[M, AsmFunction3RegionLongLongUnit](
        extracted.states,
        FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
          ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType)))),
        FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
        extracted.seqPerElt
      )

      (Some(PTypeReferenceSingleCodeType(rTyp: PStruct)), makeAnnotate) <-
        ir.CompileWithAggregators[M, AsmFunction2RegionLongLong](
          extracted.states,
          FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
          FastIndexedSeq(classInfo[Region], LongInfo),
          LongInfo,
          Let(res, extracted.results, extracted.postAggIR)
        )

      _ <- assertA(rTyp.virtualType == typ.valueType, s"$rTyp, ${typ.valueType}")

      spec = BufferSpec.blockedUncompressed
      serialize <- extracted.serialize(spec)
      deserialize <- extracted.deserialize(spec)
      combOp <- extracted.combOpFSerializedWorkersOnly(spec)

      hcl = theHailClassLoaderForSparkWorkers
      tc = ctx.taskContext
      initF = makeInit(hcl, fsBc.value, tc, ctx.r)
      globalsOffset = prev.globals.value.offset
      initAggs = ctx.r.pool.scopedRegion { aggRegion =>
        initF.newAggState(aggRegion)
        initF(ctx.r, globalsOffset)
        serialize(hcl, tc, aggRegion, initF.getAggOffset())
      }

      newRowType = PCanonicalStruct(required = true,
        localKeyPType.fields.map(f => (f.name, PType.canonical(f.typ))) ++ rTyp.fields.map(f => (f.name, f.typ)): _*)

      localBufferSize = bufferSize
      rdd = prev.rvd
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
            localBufferSize
          )
        }.aggregateByKey(initAggs, nPartitions.getOrElse(prev.rvd.getNumPartitions))(combOp, combOp)

      crdd = ContextRDD.weaken(rdd).cmapPartitionsWithIndex(
        { (i, ctx, it) =>
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
            while (i < localKeyType.size) {
              rvb.addAnnotation(localKeyType.types(i), key.get(i))
              i += 1
            }

            val aggOff = deserialize(hcl, tc, region, aggs)
            annotate.setAggState(region, aggOff)
            rvb.addAllFields(rTyp, region, annotate(region, globals))
            rvb.endStruct()
            rvb.end()
          }
        })

    } yield TableValueIntermediate(prev.copy(typ = typ,
      rvd = RVD.coerce(ctx, RVDType(newRowType, keyType.fieldNames), crdd)
    ))
}

// follows key_by non-empty key
case class TableAggregateByKey(child: TableIR, expr: IR) extends TableIR {
  require(child.typ.key.nonEmpty)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, expr)

  def copy(newChildren: IndexedSeq[BaseIR]): TableAggregateByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: TableIR, newExpr: IR) = newChildren
    TableAggregateByKey(newChild, newExpr)
  }

  val typ: TableType = child.typ.copy(rowType = child.typ.keyType ++ tcoerce[TStruct](expr.typ))

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      prev <- child.execute(r) >>= (_.asTableValue)
      prevRVD = prev.rvd.truncateKey(child.typ.key)
      ctx <- M.ask
      fsBc = ctx.fsBc
      sm = ctx.stateManager

      res = genUID()
      extracted = agg.Extract(expr, res, r.requirednessAnalysis)

      (_, makeInit) <-
        ir.CompileWithAggregators[M, AsmFunction2RegionLongUnit](
          extracted.states,
          FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
          FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
          extracted.init
        )

      (_, makeSeq) <-
        ir.CompileWithAggregators[M, AsmFunction3RegionLongLongUnit](
          extracted.states,
          FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
            ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prevRVD.rowPType)))),
          FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
          extracted.seqPerElt
        )

      valueIR = Let(res, extracted.results, extracted.postAggIR)
      keyType = prevRVD.typ.kType

      key = Ref(genUID(), keyType.virtualType)
      value = Ref(genUID(), valueIR.typ)
      (Some(PTypeReferenceSingleCodeType(rowType: PStruct)), makeRow) <-
        ir.CompileWithAggregators[M, AsmFunction3RegionLongLongLong](
          extracted.states,
          FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
            (key.name, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(keyType)))),
          FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
          Let(value.name, valueIR,
            InsertFields(key, typ.valueType.fieldNames.map(n => n -> GetField(value, n))))
        )

      _ <- assertA(rowType.virtualType == typ.rowType, s"$rowType, ${typ.rowType}")

      localChildRowType = prevRVD.rowPType
      keyIndices = prevRVD.typ.kFieldIdx
      keyOrd = prevRVD.typ.kRowOrd(ctx.stateManager)
      globalsBc <- prev.globals.broadcast

      newRVDType = prevRVD.typ.copy(rowType = rowType)

      newRVD = prevRVD
        .repartition(ctx, prevRVD.partitioner.strictify())
        .boundary
        .mapPartitionsWithIndex(newRVDType) { (i, ctx, it) =>
          val partRegion = ctx.partitionRegion
          val globalsOff = globalsBc.value.readRegionValue(partRegion, theHailClassLoaderForSparkWorkers)

          val initialize = makeInit(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), partRegion)
          val sequence = makeSeq(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), partRegion)
          val newRowF = makeRow(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), partRegion)

          val aggRegion = ctx.freshRegion()

          new Iterator[Long] {
            var isEnd = false
            var current: Long = 0
            val rowKey: WritableRegionValue = WritableRegionValue(sm, keyType, ctx.freshRegion())
            val consumerRegion: Region = ctx.region
            val newRV = RegionValue(consumerRegion)

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
                sequence(ctx.r,
                  globalsOff,
                  current)
                current = 0
              } while (hasNext && keyOrd.equiv(rowKey.value.offset, current))
              newRowF.setAggState(aggRegion, sequence.getAggOffset())

              newRowF(consumerRegion, globalsOff, rowKey.offset)
            }
          }
        }
    } yield TableValueIntermediate(prev.copy(rvd = newRVD, typ = typ))
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

  lazy val definitelyDoesNotShuffle: Boolean = TableOrderBy.isAlreadyOrdered(sortFields, child.typ.key)
  // TableOrderBy expects an unkeyed child, so that we can better optimize by
  // pushing these two steps around as needed

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableOrderBy = {
    val IndexedSeq(newChild) = newChildren
    TableOrderBy(newChild.asInstanceOf[TableIR], sortFields)
  }

  val typ: TableType = child.typ.copy(key = FastIndexedSeq())

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    child.execute(r) >>= (_.asTableValue) >>= { prev =>
      val physicalKey = prev.rvd.typ.key
      if (TableOrderBy.isAlreadyOrdered(sortFields, physicalKey))
        M.pure(TableValueIntermediate(prev.copy(typ = typ)))
      else
        M.reader { ctx =>
          val rowType = child.typ.rowType
          val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
            val i = rowType.fieldIdx(n)
            val f = rowType.fields(i)
            val fo = f.typ.ordering(ctx.stateManager)
            if (so == Ascending) fo else fo.reverse
          }.toArray

          implicit val ord: Ordering[Annotation] =
            ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

          val codec = TypedCodecSpec(prev.rvd.rowPType, BufferSpec.wireSpec)
          val rdd = prev.rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(_._1)
          val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))
          TableValueIntermediate(TableValue(typ, prev.globals, RVD.unkeyed(rowPType, orderedCRDD)))
        }
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  lazy val typ: TableType = child.typ.toTableType(entriesFieldName, colsFieldName)

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): CastMatrixToTable = {
    val IndexedSeq(newChild) = newChildren
    CastMatrixToTable(newChild.asInstanceOf[MatrixIR], entriesFieldName, colsFieldName)
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts
}

case class TableRename(child: TableIR, rowMap: Map[String, String], globalMap: Map[String, String]) extends TableIR {
  require(rowMap.keys.forall(child.typ.rowType.hasField))
  require(globalMap.keys.forall(child.typ.globalType.hasField))

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def rowF(old: String): String = rowMap.getOrElse(old, old)

  def globalF(old: String): String = globalMap.getOrElse(old, old)

  lazy val typ: TableType = child.typ.copy(
    rowType = child.typ.rowType.rename(rowMap),
    globalType = child.typ.globalType.rename(globalMap),
    key = child.typ.key.map(k => rowMap.getOrElse(k, k))
  )

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableRename = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRename(newChild, rowMap, globalMap)
  }

  protected[ir] override def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    for {tv <- child.execute(r) >>= (_.asTableValue)}
      yield TableValueIntermediate(tv.rename(globalMap, rowMap))
}

case class TableFilterIntervals(child: TableIR, intervals: IndexedSeq[Interval], keep: Boolean) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableFilterIntervals(newChild, intervals, keep)
  }

  override lazy val typ: TableType = child.typ

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {
      tv <- child.execute(r) >>= (_.asTableValue)
      partitioner <- M.reader { ctx =>
        RVDPartitioner.union(
          ctx.stateManager,
          tv.typ.keyType,
          intervals,
          tv.typ.keyType.size - 1)
      }
    } yield TableValueIntermediate(
      TableValue(tv.typ, tv.globals, tv.rvd.filterIntervals(partitioner, keep))
    )
}

case class MatrixToTableApply(child: MatrixIR, function: MatrixToTableFunction) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = if (function.preservesPartitionCounts) child.rowCountUpperBound else None

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None
}

case class TableToTableApply(child: TableIR, function: TableToTableFunction) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None

  lazy val rowCountUpperBound: Option[Long] = if (function.preservesPartitionCounts) child.rowCountUpperBound else None

  protected[ir] override def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
  for {tv <- child.execute(r) >>= (_.asTableValue); fv <- function.execute(tv)}
    yield TableValueIntermediate(fv)

}

case class BlockMatrixToTableApply(
  bm: BlockMatrixIR,
  aux: IR,
  function: BlockMatrixToTableFunction) extends TableIR {

  override lazy val childrenSeq: IndexedSeq[BaseIR] = Array(bm, aux)

  lazy val rowCountUpperBound: Option[Long] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR =
    BlockMatrixToTableApply(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[IR],
      function)

  override lazy val typ: TableType = function.typ(bm.typ, aux.typ)

  override protected[ir] def execute[M[_]: MonadLower](r: LoweringAnalyses): M[TableExecuteIntermediate] =
    for {
      b <- bm.execute
      a <- CompileAndEvaluate[M, Annotation](aux, optimize = false)
      tv <- function.execute(b, a)
  } yield TableValueIntermediate(tv)
}

case class BlockMatrixToTable(child: BlockMatrixIR) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixToTable(newChild)
  }

  override val typ: TableType = {
    val rvType = TStruct("i" -> TInt64, "j" -> TInt64, "entry" -> TFloat64)
    TableType(rvType, Array[String](), TStruct.empty)
  }

  override protected[ir] def execute[M[_]](r: LoweringAnalyses)(implicit M: MonadLower[M]): M[TableExecuteIntermediate] =
    for {bm <- child.execute; ctx <- M.ask}
        yield TableValueIntermediate(bm.entriesTable(ctx))
}

case class RelationalLetTable(name: String, value: IR, body: TableIR) extends TableIR {
  def typ: TableType = body.typ

  lazy val rowCountUpperBound: Option[Long] = body.rowCountUpperBound

  def childrenSeq: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newValue: IR, newBody: TableIR) = newChildren
    RelationalLetTable(name, newValue, newBody)
  }
}
