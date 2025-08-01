package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailStateManager, HailTaskContext, TaskFinalizer}
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.{
  BlockMatrixToTableFunction, IntervalFunctions, MatrixToTableFunction, TableToTableFunction,
}
import is.hail.expr.ir.lowering._
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

import java.io.{Closeable, InputStream}

import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods

object TableIR {
  def read(fs: FS, path: String, dropRows: Boolean = false, requestedType: Option[TableType] = None)
    : TableRead = {
    val successFile = path + "/_SUCCESS"
    if (!fs.isFile(path + "/_SUCCESS"))
      fatal(s"write failed: file not found: $successFile")

    val tr = TableNativeReader.read(fs, path, None)
    TableRead(requestedType.getOrElse(tr.fullType), dropRows = dropRows, tr)
  }

  val globalName: Name = Name("global")

  val rowName: Name = Name("row")
}

sealed abstract class TableIR extends BaseIR {
  def typ: TableType

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR
}

object TableLiteral {
  def apply(value: TableValue, theHailClassLoader: HailClassLoader): TableLiteral =
    TableLiteral(
      value.typ,
      value.rvd,
      value.globals.encoding,
      value.globals.encodeToByteArrays(theHailClassLoader),
    )
}

case class TableLiteral(
  typ: TableType,
  rvd: RVD,
  enc: AbstractTypedCodecSpec,
  encodedGlobals: Array[Array[Byte]],
) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    TableLiteral(typ, rvd, enc, encodedGlobals)
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

  type LoweredTableReaderCoercer =
    (ExecuteContext, IR, Type, IndexedSeq[Any], IR => IR) => TableStage

  def makeCoercer(
    ctx: ExecuteContext,
    key: IndexedSeq[String],
    partitionKey: Int,
    uidFieldName: String,
    contextType: TStruct,
    contexts: IndexedSeq[Any],
    keyType: TStruct,
    bodyPType: TStruct => PStruct,
    keys: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long],
    context: String,
  ): LoweredTableReaderCoercer = {
    assert(key.nonEmpty)
    assert(contexts.nonEmpty)
    assert(contextType.hasField("partitionIndex"))
    assert(contextType.fieldType("partitionIndex") == TInt32)

    val nPartitions = contexts.length
    val sampleSize = math.min(nPartitions * 20, 1000000)
    val samplesPerPartition = sampleSize / nPartitions

    val pkType = keyType.typeAfterSelectNames(key.take(partitionKey))

    def selectPK(k: IR): IR =
      SelectFields(k, key.take(partitionKey))

    info(s"scanning $context for sortedness...")
    val prevkey = AggSignature(PrevNonnull(), FastSeq(), FastSeq(keyType))
    val count = AggSignature(Count(), FastSeq(), FastSeq())
    val samplekey = AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(keyType, TFloat64))
    val sum = AggSignature(Sum(), FastSeq(), FastSeq(TInt64))
    val minkey = AggSignature(TakeBy(), FastSeq(TInt32), FastSeq(keyType, keyType))
    val maxkey = AggSignature(TakeBy(Descending), FastSeq(TInt32), FastSeq(keyType, keyType))

    val xType = TStruct(
      "key" -> keyType,
      "token" -> TFloat64,
      "prevkey" -> keyType,
    )

    val keyRef = Ref(freshName(), keyType)
    val xRef = Ref(freshName(), xType)
    val nRef = Ref(freshName(), TInt64)

    val scanBody = (ctx: IR) =>
      StreamAgg(
        StreamAggScan(
          ReadPartition(
            ctx,
            keyType,
            new PartitionIteratorLongReader(
              keyType,
              uidFieldName,
              contextType,
              (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
              (requestedType: Type) => keys(requestedType.asInstanceOf[TStruct]),
            ),
          ),
          keyRef.name,
          MakeStruct(FastSeq(
            "key" -> keyRef,
            "token" -> invokeSeeded(
              "rand_unif",
              1,
              TFloat64,
              RNGStateLiteral(),
              F64(0.0),
              F64(1.0),
            ),
            "prevkey" -> ApplyScanOp(FastSeq(), FastSeq(keyRef), prevkey),
          )),
        ),
        xRef.name,
        Let(
          FastSeq(nRef.name -> ApplyAggOp(FastSeq(), FastSeq(), count)),
          AggLet(
            keyRef.name,
            GetField(xRef, "key"),
            MakeStruct(FastSeq(
              "n" -> nRef,
              "minkey" ->
                ApplyAggOp(
                  FastSeq(I32(1)),
                  FastSeq(keyRef, keyRef),
                  minkey,
                ),
              "maxkey" ->
                ApplyAggOp(
                  FastSeq(I32(1)),
                  FastSeq(keyRef, keyRef),
                  maxkey,
                ),
              "ksorted" ->
                ApplyComparisonOp(
                  EQ(TInt64),
                  ApplyAggOp(
                    FastSeq(),
                    FastSeq(
                      invoke(
                        "toInt64",
                        TInt64,
                        invoke(
                          "lor",
                          TBoolean,
                          IsNA(GetField(xRef, "prevkey")),
                          ApplyComparisonOp(
                            LTEQ(keyType),
                            GetField(xRef, "prevkey"),
                            GetField(xRef, "key"),
                          ),
                        ),
                      )
                    ),
                    sum,
                  ),
                  nRef,
                ),
              "pksorted" ->
                ApplyComparisonOp(
                  EQ(TInt64),
                  ApplyAggOp(
                    FastSeq(),
                    FastSeq(
                      invoke(
                        "toInt64",
                        TInt64,
                        invoke(
                          "lor",
                          TBoolean,
                          IsNA(selectPK(GetField(xRef, "prevkey"))),
                          ApplyComparisonOp(
                            LTEQ(pkType),
                            selectPK(GetField(xRef, "prevkey")),
                            selectPK(GetField(xRef, "key")),
                          ),
                        ),
                      )
                    ),
                    sum,
                  ),
                  nRef,
                ),
              "sample" -> ApplyAggOp(
                FastSeq(I32(samplesPerPartition)),
                FastSeq(GetField(xRef, "key"), GetField(xRef, "token")),
                samplekey,
              ),
            )),
            isScan = false,
          ),
        ),
      )

    val scanResult = cdaIR(
      ToStream(Literal(TArray(contextType), contexts)),
      MakeStruct(FastSeq()),
      "table_coerce_sortedness",
      NA(TString),
    )((context, _) => scanBody(context))

    val sortedPartDataIR = sortIR(bindIR(scanResult) { scanResult =>
      mapIR(
        filterIR(
          mapIR(
            rangeIR(I32(0), ArrayLen(scanResult))
          ) { i =>
            InsertFields(
              ArrayRef(scanResult, i),
              FastSeq("i" -> i),
            )
          }
        )(row => ArrayLen(GetField(row, "minkey")) > 0)
      ) { row =>
        InsertFields(
          row,
          FastSeq(
            ("minkey", ArrayRef(GetField(row, "minkey"), I32(0))),
            ("maxkey", ArrayRef(GetField(row, "maxkey"), I32(0))),
          ),
        )
      }
    }) { (l, r) =>
      ApplyComparisonOp(
        LT(TStruct("minkey" -> keyType, "maxkey" -> keyType)),
        SelectFields(l, FastSeq("minkey", "maxkey")),
        SelectFields(r, FastSeq("minkey", "maxkey")),
      )
    }

    val summary = bindIR(sortedPartDataIR) { sortedPartData =>
      MakeStruct(FastSeq(
        "ksorted" ->
          invoke(
            "land",
            TBoolean,
            foldIR(ToStream(sortedPartData), True()) { (acc, partDataWithIndex) =>
              invoke("land", TBoolean, acc, GetField(partDataWithIndex, "ksorted"))
            },
            foldIR(StreamRange(I32(0), ArrayLen(sortedPartData) - I32(1), I32(1)), True()) {
              (acc, i) =>
                invoke(
                  "land",
                  TBoolean,
                  acc,
                  ApplyComparisonOp(
                    LTEQ(keyType),
                    GetField(ArrayRef(sortedPartData, i), "maxkey"),
                    GetField(ArrayRef(sortedPartData, i + I32(1)), "minkey"),
                  ),
                )
            },
          ),
        "pksorted" ->
          invoke(
            "land",
            TBoolean,
            foldIR(ToStream(sortedPartData), True()) { (acc, partDataWithIndex) =>
              invoke("land", TBoolean, acc, GetField(partDataWithIndex, "pksorted"))
            },
            foldIR(StreamRange(I32(0), ArrayLen(sortedPartData) - I32(1), I32(1)), True()) {
              (acc, i) =>
                invoke(
                  "land",
                  TBoolean,
                  acc,
                  ApplyComparisonOp(
                    LTEQ(pkType),
                    selectPK(GetField(ArrayRef(sortedPartData, i), "maxkey")),
                    selectPK(GetField(ArrayRef(sortedPartData, i + I32(1)), "minkey")),
                  ),
                )
            },
          ),
        "sortedPartData" -> sortedPartData,
      ))
    }

    val (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) =
      Compile[AsmFunction1RegionLong](
        ctx,
        FastSeq(),
        FastSeq[TypeInfo[_]](classInfo[Region]),
        LongInfo,
        summary,
        optimize = true,
      )

    val s = ctx.scopedExecution { (hcl, fs, htc, r) =>
      val a = f(hcl, fs, htc, r)(r)
      SafeRow(resultPType, a)
    }

    val ksorted = s.getBoolean(0)
    val pksorted = s.getBoolean(1)
    val sortedPartData = s.getAs[IndexedSeq[Row]](2)

    if (ksorted) {
      info(s"Coerced sorted $context - no additional import work to do")
      (
        ctx: ExecuteContext,
        globals: IR,
        contextType: Type,
        contexts: IndexedSeq[Any],
        body: IR => IR,
      ) => {
        val partOrigIndex = sortedPartData.map(_.getInt(6))

        val partitioner = new RVDPartitioner(
          ctx.stateManager,
          keyType,
          sortedPartData.map { partData =>
            Interval(
              partData.get(1),
              partData.get(2),
              includesStart = true,
              includesEnd = true,
            )
          },
          key.length,
        )

        TableStage(
          globals,
          partitioner,
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
          body,
        )
      }
    } else if (pksorted) {
      info(
        s"Coerced prefix-sorted $context, requiring additional sorting within data partitions on each query."
      )

      def selectPK(r: Row): Row = {
        val a = new Array[Any](partitionKey)
        var i = 0
        while (i < partitionKey) {
          a(i) = r.get(i)
          i += 1
        }
        Row.fromSeq(a)
      }

      (
        ctx: ExecuteContext,
        globals: IR,
        contextType: Type,
        contexts: IndexedSeq[Any],
        body: IR => IR,
      ) => {
        val partOrigIndex = sortedPartData.map(_.getInt(6))

        val partitioner = new RVDPartitioner(
          ctx.stateManager,
          pkType,
          sortedPartData.map { partData =>
            Interval(
              selectPK(partData.getAs[Row](1)),
              selectPK(partData.getAs[Row](2)),
              includesStart = true,
              includesEnd = true,
            )
          },
          pkType.size,
        )

        val pkPartitioned = TableStage(
          globals,
          partitioner,
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
          body,
        )

        pkPartitioned
          .extendKeyPreservesPartitioning(ctx, key)
          .mapPartition(None) { part =>
            flatMapIR(StreamGroupByKey(part, pkType.fieldNames, missingEqual = true)) {
              inner => ToStream(sortIR(inner) { case (l, r) => ApplyComparisonOp(LT(l.typ), l, r) })
            }
          }
      }
    } else {
      info(
        s"$context is out of order..." +
          s"\n  Write the dataset to disk before running multiple queries to avoid multiple costly data shuffles."
      )

      (
        ctx: ExecuteContext,
        globals: IR,
        contextType: Type,
        contexts: IndexedSeq[Any],
        body: IR => IR,
      ) => {
        val partOrigIndex = sortedPartData.map(_.getInt(6))

        val partitioner = RVDPartitioner.unkeyed(ctx.stateManager, sortedPartData.length)

        val tableStage = TableStage(
          globals,
          partitioner,
          TableStageDependency.none,
          ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
          body,
        )

        val rowRType =
          VirtualTypeWithReq(bodyPType(tableStage.rowType)).r.asInstanceOf[RStruct]
        val globReq = Requiredness(globals, ctx)
        val globRType = globReq.lookup(globals).asInstanceOf[RStruct]

        ctx.backend.lowerDistributedSort(
          ctx,
          tableStage,
          keyType.fieldNames.map(f => SortField(f, Ascending)),
          RTable(rowRType, globRType, FastSeq()),
        ).lower(
          ctx,
          TableType(tableStage.rowType, keyType.fieldNames, globals.typ.asInstanceOf[TStruct]),
        )
      }
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
        Array((uidFieldName, uidType))
      )
    )
  }

  def uidType: Type

  protected def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq

  protected def uidRequiredness: VirtualTypeWithReq

  override def rowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq = {
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
        RStruct(concreteRFields :+ RField(uidFieldName, uidRequiredness.r, concreteRFields.length)),
      )
    } else {
      concreteRowReq
    }
  }
}

abstract class TableReader {
  def pathsUsed: Seq[String]

  final def lower(ctx: ExecuteContext, requestedType: TableType, dropRows: Boolean): TableStage = {
    if (dropRows) {
      val globals = lowerGlobals(ctx, requestedType.globalType)

      TableStage(
        globals,
        RVDPartitioner.empty(ctx, requestedType.keyType),
        TableStageDependency.none,
        MakeStream(FastSeq(), TStream(TStruct.empty)),
        (_: Ref) => MakeStream(FastSeq(), TStream(requestedType.rowType)),
      )
    } else {
      lower(ctx, requestedType)
    }
  }

  def toExecuteIntermediate(ctx: ExecuteContext, requestedType: TableType, dropRows: Boolean)
    : TableExecuteIntermediate =
    TableExecuteIntermediate(lower(ctx, requestedType, dropRows))

  def partitionCounts: Option[IndexedSeq[Long]]

  def isDistinctlyKeyed: Boolean = false // FIXME: No default value

  def fullType: TableType

  def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq

  def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq

  def toJValue: JValue =
    Extraction.decompose(this)(TableReader.formats)

  def renderShort(): String

  def defaultRender(): String =
    StringEscapeUtils.escapeString(JsonMethods.compact(toJValue))

  def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR

  def lower(ctx: ExecuteContext, requestedType: TableType): TableStage
}

object TableNativeReader {
  def read(fs: FS, path: String, options: Option[NativeReaderOptions]): TableNativeReader =
    TableNativeReader(fs, TableNativeReaderParameters(path, options))

  def apply(fs: FS, params: TableNativeReaderParameters): TableNativeReader = {
    val spec = (RelationalSpec.read(fs, params.path): @unchecked) match {
      case ts: AbstractTableSpec => ts
      case _: AbstractMatrixTableSpec =>
        fatal(s"file is a MatrixTable, not a Table: '${params.path}'")
    }

    val filterIntervals = params.options.map(_.filterIntervals).getOrElse(false)

    if (filterIntervals && !spec.indexed)
      fatal(
        """`intervals` specified on an unindexed table.
          |This table was written using an older version of hail
          |rewrite the table in order to create an index to proceed""".stripMargin
      )

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

  override def fullRowType: TStruct =
    rvd.rowType.insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64)))

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
    requestedType: TStruct,
  ): IEmitCode = {
    val eltRef = Ref(freshName(), rvd.rowType)

    val (Some(PTypeReferenceSingleCodeType(upcastPType: PBaseStruct)), upcast) =
      Compile[AsmFunction2RegionLongLong](
        ctx,
        FastSeq((
          eltRef.name,
          SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)),
        )),
        FastSeq(classInfo[Region], LongInfo),
        LongInfo,
        PruneDeadFields.upcast(ctx, eltRef, requestedType),
      )

    val upcastCode = mb.getObject[Function4[
      HailClassLoader,
      FS,
      HailTaskContext,
      Region,
      AsmFunction2RegionLongLong,
    ]](upcast)

    val rowPType = rvd.rowPType.subsetTo(requestedType)

    val createUID = requestedType.hasField(uidFieldName)

    assert(
      upcastPType == rowPType,
      s"ptype mismatch:\n  upcast: $upcastPType\n  computed: $rowPType\n  inputType: ${rvd.rowPType}\n  requested: $requestedType",
    )

    context.toI(cb).map(cb) { _partIdx =>
      val partIdx = cb.memoizeField(_partIdx, "partIdx")
      val iterator = mb.genFieldThisRef[Iterator[Long]]("rvdreader_iterator")
      val next = mb.genFieldThisRef[Long]("rvdreader_next")
      val curIdx = mb.genFieldThisRef[Long]("rvdreader_curIdx")

      val region = mb.genFieldThisRef[Region]("rvdreader_region")
      val upcastF = mb.genFieldThisRef[AsmFunction2RegionLongLong]("rvdreader_upcast")

      val broadcastRVD =
        mb.getObject[BroadcastRVD](new BroadcastRVD(ctx.backend.asSpark, rvd))

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          cb.assign(curIdx, 0L)
          cb.assign(
            iterator,
            broadcastRVD.invoke[Int, Region, Region, Iterator[Long]](
              "computePartition",
              partIdx.asInt.value,
              region,
              partitionRegion,
            ),
          )
          cb.assign(
            upcastF,
            Code.checkcast[AsmFunction2RegionLongLong](upcastCode.invoke[
              AnyRef,
              AnyRef,
              AnyRef,
              AnyRef,
              AnyRef,
            ](
              "apply",
              cb.emb.ecb.emodb.getHailClassLoader,
              cb.emb.ecb.emodb.getFS,
              cb.emb.ecb.getTaskContext,
              partitionRegion,
            )),
          )
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(!iterator.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(curIdx, curIdx + 1)
          cb.assign(
            next,
            upcastF.invoke[Region, Long, Long](
              "apply",
              region,
              Code.longValue(iterator.invoke[java.lang.Long]("next")),
            ),
          )
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          if (createUID) {
            val uid = SStackStruct.constructFromArgs(
              cb,
              region,
              TTuple(TInt64, TInt64),
              EmitCode.present(mb, partIdx),
              EmitCode.present(mb, primitive(cb.memoize(curIdx - 1))),
            )
            IEmitCode.present(
              cb,
              upcastPType.loadCheapSCode(cb, next)
                ._insert(requestedType, uidFieldName -> EmitValue.present(uid)),
            )
          } else {
            IEmitCode.present(cb, upcastPType.loadCheapSCode(cb, next))
          }
        }

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamValue(producer)
    }
  }

  def toJValue: JValue =
    JString("<PartitionRVDReader>") // cannot be parsed, but need a printout for Pretty
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
    requestedType: TStruct,
  ): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(
      uidFieldName
    ) && !spec.encodedVirtualType.asInstanceOf[TStruct].hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType

    val concreteSType = spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStruct]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)),
    )
    val elementSType = if (insertUID)
      SInsertFieldsStruct(
        requestedType,
        concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)),
      )
    else
      concreteSType

    context.toI(cb).map(cb) { case ctxStruct: SBaseStructValue =>
      val partIdx =
        cb.memoizeField(ctxStruct.loadField(cb, "partitionIndex").getOrAssert(cb), "partIdx")
      val rowIdx = mb.genFieldThisRef[Long]("pnr_rowidx")
      val pathString =
        cb.memoizeField(
          ctxStruct.loadField(cb, "partitionPath").getOrAssert(cb).asString.loadString(cb)
        )
      val xRowBuf = mb.genFieldThisRef[InputBuffer]("pnr_xrowbuf")
      val next = mb.newPSettable(mb.fieldBuilder, elementSType, "pnr_next")
      val region = mb.genFieldThisRef[Region]("pnr_region")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          cb.assign(
            xRowBuf,
            spec.buildCodeInputBuffer(mb.openUnbuffered(pathString, checkCodec = true)),
          )
          cb.assign(rowIdx, -1L)
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(!xRowBuf.readByte().toZ, cb.goto(LendOfStream))

          val base = spec.encodedType.buildDecoder(concreteType, cb.emb.ecb).apply(
            cb,
            region,
            xRowBuf,
          ).asBaseStruct
          if (insertUID) {
            cb.assign(rowIdx, rowIdx + 1)
            val uid = EmitValue.present(
              new SStackStructValue(
                uidSType,
                Array(
                  EmitValue.present(partIdx),
                  EmitValue.present(new SInt64Value(rowIdx)),
                ),
              )
            )
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

case class PartitionNativeIntervalReader(
  sm: HailStateManager,
  tablePath: String,
  tableSpec: AbstractTableSpec,
  uidFieldName: String,
) extends AbstractNativeReader {
  require(tableSpec.indexed)

  lazy val rowsSpec = tableSpec.rowsSpec
  lazy val spec = rowsSpec.typedCodecSpec
  lazy val indexSpec = tableSpec.rowsSpec.asInstanceOf[Indexed].indexSpec
  lazy val partitioner = rowsSpec.partitioner(sm)

  lazy val contextType: Type = RVDPartitioner.intervalIRRepresentation(partitioner.kType)
  require(partitioner.kType.size > 0)

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType
    val concreteSType: SBaseStruct =
      spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStruct]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)),
    )
    val eltSType: SBaseStruct = if (insertUID)
      SInsertFieldsStruct(
        requestedType,
        concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)),
      )
    else
      concreteSType

    val index = new StagedIndexReader(cb.emb, indexSpec)

    context.toI(cb).map(cb) { case _ctx: SIntervalValue =>
      val ctx = cb.memoizeField(_ctx, "ctx").asInterval

      val partitionerLit = partitioner.partitionBoundsIRRepresentation
      val partitionerRuntime = cb.emb.addLiteral(
        cb,
        partitionerLit.value,
        VirtualTypeWithReq.fullyOptional(partitionerLit.typ),
      )
        .asIndexable

      val pathsType = VirtualTypeWithReq.fullyRequired(TArray(TString))
      val rowsPath = tableSpec.rowsComponent.absolutePath(tablePath)
      val partitionPathsRuntime = cb.memoizeField(
        mb.addLiteral(cb, rowsSpec.absolutePartPaths(rowsPath).toFastSeq, pathsType),
        "partitionPathsRuntime",
      )
        .asIndexable
      val indexPathsRuntime = cb.memoizeField(
        mb.addLiteral(
          cb,
          rowsSpec.partFiles.map(partPath =>
            s"$rowsPath/${indexSpec.relPath}/$partPath.idx"
          ).toFastSeq,
          pathsType,
        ),
        "indexPathsRuntime",
      )
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

          val startBound = ctx.loadStart(cb).getOrAssert(cb)
          val includesStart = ctx.includesStart
          val endBound = ctx.loadEnd(cb).getOrAssert(cb)
          val includesEnd = ctx.includesEnd

          val (startPart, endPart) = IntervalFunctions.partitionerFindIntervalRange(
            cb,
            partitionerRuntime,
            SStackInterval.construct(
              EmitValue.present(startBound),
              EmitValue.present(endBound),
              includesStart,
              includesEnd,
            ),
            -1,
          )

          cb.if_(
            endPart < startPart,
            cb._fatal(
              "invalid start/end config - startPartIdx=",
              startPartitionIndex.toS,
              ", endPartIdx=",
              lastIncludedPartitionIdx.toS,
            ),
          )

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
          cb.if_(
            currIdxInPartition >= stopIdxInPartition, {
              cb.if_(
                currPartitionIdx >= partitioner.numPartitions || currPartitionIdx > lastIncludedPartitionIdx,
                cb.goto(LendOfStream),
              )

              val requiresIndexInit = cb.newLocal[Boolean]("requiresIndexInit")

              cb.if_(
                streamFirst,
                // if first, reuse open index from previous time the stream was run if possible
                // this is a common case if looking up nearby keys
                cb.assign(
                  requiresIndexInit,
                  !(indexInitialized && (indexCachedIndex ceq currPartitionIdx)),
                ), {
                  /* if not first, then the index must be open to the previous partition and needs
                   * to be reinitialized */
                  cb.assign(streamFirst, false)
                  cb.assign(requiresIndexInit, true)
                },
              )

              cb.if_(
                requiresIndexInit, {
                  cb.if_(
                    indexInitialized, {
                      cb += finalizer.invoke[Unit]("clear")
                      index.close(cb)
                      cb += ib.close()
                    },
                    cb.assign(indexInitialized, true),
                  )
                  cb.assign(indexCachedIndex, currPartitionIdx)
                  val partPath =
                    partitionPathsRuntime.loadElement(cb, currPartitionIdx).getOrAssert(
                      cb
                    ).asString.loadString(cb)
                  val idxPath = indexPathsRuntime.loadElement(cb, currPartitionIdx).getOrAssert(
                    cb
                  ).asString.loadString(cb)
                  index.initialize(cb, idxPath)
                  cb.assign(
                    ib,
                    spec.buildCodeInputBuffer(
                      Code.newInstance[ByteTrackingInputStream, InputStream](
                        cb.emb.openUnbuffered(partPath, false)
                      )
                    ),
                  )
                  index.addToFinalizer(cb, finalizer)
                  cb += finalizer.invoke[Closeable, Unit]("addCloseable", ib)
                },
              )

              cb.if_(
                currPartitionIdx ceq lastIncludedPartitionIdx, {
                  cb.if_(
                    currPartitionIdx ceq startPartitionIndex, {
                      // query the full interval
                      val indexResult = index.queryInterval(cb, ctx)
                      val startIdx = indexResult.loadField(cb, 0)
                        .getOrAssert(cb)
                        .asInt64
                        .value
                      cb.assign(currIdxInPartition, startIdx)
                      val endIdx = indexResult.loadField(cb, 1)
                        .getOrAssert(cb)
                        .asInt64
                        .value
                      cb.assign(stopIdxInPartition, endIdx)
                      cb.if_(
                        endIdx > startIdx, {
                          val leafNode = indexResult.loadField(cb, 2).getOrAssert(cb).asBaseStruct
                          val firstOffset = index.offsetAnnotation(cb, leafNode)
                          cb += ib.seek(firstOffset.asInt64.value)
                        },
                      )
                    }, {
                      // read from start of partition to the end interval

                      val indexResult =
                        index.queryBound(
                          cb,
                          ctx.loadEnd(cb).getOrAssert(cb).asBaseStruct,
                          ctx.includesEnd,
                        )
                      val startIdx = indexResult.loadField(cb, 0).getOrAssert(cb).asInt64.value
                      cb.assign(currIdxInPartition, 0L)
                      cb.assign(stopIdxInPartition, startIdx)
                      // no need to seek, starting at beginning of partition
                    },
                  )
                }, {
                  cb.if_(
                    currPartitionIdx ceq startPartitionIndex, {
                      // read from left endpoint until end of partition
                      val indexResult = index.queryBound(
                        cb,
                        ctx.loadStart(cb).getOrAssert(cb).asBaseStruct,
                        cb.memoize(!ctx.includesStart),
                      )
                      val startIdx = indexResult.loadField(cb, 0).getOrAssert(cb).asInt64.value

                      cb.assign(currIdxInPartition, startIdx)
                      cb.assign(stopIdxInPartition, index.nKeys(cb))
                      cb.if_(
                        currIdxInPartition < stopIdxInPartition, {
                          val leafNode = indexResult.loadField(cb, 1).getOrAssert(cb).asBaseStruct
                          val firstOffset = index.offsetAnnotation(cb, leafNode)
                          cb += ib.seek(firstOffset.asInt64.value)
                        },
                      )
                    }, {
                      // in the middle of a partition run, so read everything
                      cb.assign(currIdxInPartition, 0L)
                      cb.assign(stopIdxInPartition, index.nKeys(cb))
                    },
                  )
                },
              )

              cb.assign(currPartitionIdx, currPartitionIdx + 1)
              cb.goto(Lstart)
            },
          )

          cb.if_(ib.readByte() cne 1, cb._fatal(s"bad buffer state!"))
          cb.assign(currIdxInPartition, currIdxInPartition + 1L)
          val decRow =
            spec.encodedType.buildDecoder(requestedType, cb.emb.ecb)(cb, region, ib).asBaseStruct
          cb.assign(
            decodedRow,
            if (insertUID)
              decRow.insert(
                cb,
                elementRegion,
                eltSType.virtualType.asInstanceOf[TStruct],
                uidFieldName -> EmitValue.present(uidSType.fromEmitCodes(
                  cb,
                  FastSeq(
                    EmitCode.present(mb, primitive(currPartitionIdx)),
                    EmitCode.present(mb, primitive(currIdxInPartition)),
                  ),
                )),
              )
            else decRow,
          )
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
  uidFieldName: String,
) extends AbstractNativeReader {
  def contextType: Type = TStruct(
    "partitionIndex" -> TInt64,
    "partitionPath" -> TString,
    "indexPath" -> TString,
    "interval" -> RVDPartitioner.intervalIRRepresentation(
      spec.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
    ),
  )

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName)
    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType
    val concreteSType: SBaseStructPointer =
      spec.encodedType.decodedSType(concreteType).asInstanceOf[SBaseStructPointer]
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)),
    )
    val eltSType: SBaseStruct = if (insertUID)
      SInsertFieldsStruct(
        requestedType,
        concreteSType,
        Array(uidFieldName -> EmitType(uidSType, true)),
      )
    else
      concreteSType
    val index = new StagedIndexReader(cb.emb, indexSpec)

    context.toI(cb).map(cb) { case ctxStruct: SBaseStructValue =>
      val partIdx =
        cb.memoizeField(ctxStruct.loadField(cb, "partitionIndex").getOrAssert(cb), "partIdx")
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
            .getOrAssert(cb)
            .asString
            .loadString(cb)
          val partitionPath = ctxStruct
            .loadField(cb, "partitionPath")
            .getOrAssert(cb)
            .asString
            .loadString(cb)
          val interval = ctxStruct
            .loadField(cb, "interval")
            .getOrAssert(cb)
            .asInterval
          index.initialize(cb, indexPath)

          val indexResult = index.queryInterval(cb, interval)
          val startIndex = indexResult.loadField(cb, 0)
            .getOrAssert(cb)
            .asInt64
            .value
          val endIndex = indexResult.loadField(cb, 1)
            .getOrAssert(cb)
            .asInt64
            .value
          cb.assign(curIdx, startIndex)
          cb.assign(endIdx, endIndex)

          cb.assign(
            ib,
            spec.buildCodeInputBuffer(
              Code.newInstance[ByteTrackingInputStream, InputStream](
                cb.emb.openUnbuffered(partitionPath, false)
              )
            ),
          )
          cb.if_(
            endIndex > startIndex, {
              val leafNode = indexResult.loadField(cb, 2).getOrAssert(cb).asBaseStruct
              val firstOffset = index.offsetAnnotation(cb, leafNode)
              cb += ib.seek(firstOffset.asInt64.value)
            },
          )
          index.close(cb)
        }
        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(curIdx >= endIdx, cb.goto(LendOfStream))
          val next = ib.readByte()
          cb.if_(next cne 1, cb._fatal(s"bad buffer state!"))
          val base =
            spec.encodedType.buildDecoder(concreteType, cb.emb.ecb)(cb, region, ib).asBaseStruct
          if (insertUID)
            cb.assign(
              decodedRow,
              new SInsertFieldsStructValue(
                eltSType.asInstanceOf[SInsertFieldsStruct],
                base,
                Array(EmitValue.present(
                  new SStackStructValue(
                    uidSType,
                    Array(
                      EmitValue.present(partIdx),
                      EmitValue.present(primitive(curIdx)),
                    ),
                  )
                )),
              ),
            )
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
    "rightContext" -> right.contextType,
  )

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

    RStruct.fromNamesAndTypes(requestedType.fieldNames.map(f =>
      (f, lRequired.fieldType.getOrElse(f, rRequired.fieldType(f)))
    ))
  }

  lazy val fullRowType: TStruct = {
    val leftStruct = left.fullRowType.deleteKey(left.uidFieldName)
    val rightStruct = right.fullRowType
    TStruct.concat(leftStruct, rightStruct)
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
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

case class PartitionZippedNativeIntervalReader(
  sm: HailStateManager,
  mtPath: String,
  mtSpec: AbstractMatrixTableSpec,
  uidFieldName: String,
) extends PartitionReader {
  require(mtSpec.indexed)

  private[this] class PartitionEntriesNativeIntervalReader(
    sm: HailStateManager,
    entriesPath: String,
    entriesSpec: AbstractTableSpec,
    uidFieldName: String,
    rowsTableSpec: AbstractTableSpec,
  ) extends PartitionNativeIntervalReader(sm, entriesPath, entriesSpec, uidFieldName) {
    override lazy val partitioner = rowsTableSpec.rowsSpec.partitioner(sm)
  }

  // XXX: rows and entries paths are hardcoded, see MatrixTableSpec
  private lazy val rowsReader =
    PartitionNativeIntervalReader(sm, mtPath + "/rows", mtSpec.rowsSpec, "__dummy")

  private lazy val entriesReader =
    new PartitionEntriesNativeIntervalReader(
      sm,
      mtPath + "/entries",
      mtSpec.entriesSpec,
      uidFieldName,
      rowsReader.tableSpec,
    )

  private lazy val zippedReader = PartitionZippedNativeReader(rowsReader, entriesReader)

  def contextType = rowsReader.contextType
  def fullRowType = zippedReader.fullRowType
  def rowRequiredness(requestedType: TStruct): RStruct = zippedReader.rowRequiredness(requestedType)
  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    codeContext: EmitCode,
    requestedType: TStruct,
  ): IEmitCode = {
    val zipContextType: TBaseStruct = tcoerce(zippedReader.contextType)
    val valueContext = cb.memoize(codeContext)
    val contexts: IndexedSeq[EmitCode] = FastSeq(valueContext, valueContext)
    val st = SStackStruct(zipContextType, contexts.map(_.emitType))
    val context = EmitCode.present(mb, st.fromEmitCodes(cb, contexts))

    zippedReader.emitStream(ctx, cb, mb, context, requestedType)
  }
}

case class PartitionZippedIndexedNativeReader(
  specLeft: AbstractTypedCodecSpec,
  specRight: AbstractTypedCodecSpec,
  indexSpecLeft: AbstractIndexSpec,
  indexSpecRight: AbstractIndexSpec,
  key: IndexedSeq[String],
  uidFieldName: String,
) extends PartitionReader {

  def contextType: Type = {
    TStruct(
      "partitionIndex" -> TInt64,
      "leftPartitionPath" -> TString,
      "rightPartitionPath" -> TString,
      "indexPath" -> TString,
      "interval" -> RVDPartitioner.intervalIRRepresentation(
        specLeft.encodedVirtualType.asInstanceOf[TStruct].select(key)._1
      ),
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
    val pt = specLeft.decodedPType(leftStruct).asInstanceOf[PStruct].insertFields(
      specRight.decodedPType(rightStruct).asInstanceOf[PStruct].fields.map(f => (f.name, f.typ))
    )
    rt.fromPType(pt)
    rt
  }

  val uidSType: SStackStruct = SStackStruct(
    TTuple(TInt64, TInt64, TInt64, TInt64),
    Array(
      EmitType(SInt64, true),
      EmitType(SInt64, true),
      EmitType(SInt64, true),
      EmitType(SInt64, true),
    ),
  )

  def fullRowType: TStruct =
    (specLeft.encodedVirtualType.asInstanceOf[TStruct] ++ specRight.encodedVirtualType.asInstanceOf[
      TStruct
    ])
      .insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64, TInt64, TInt64)))

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct,
  ): IEmitCode = {

    val (leftRType, rightRType) = splitRequestedTypes(requestedType)

    val insertUID: Boolean = requestedType.hasField(uidFieldName)

    val index = new StagedIndexReader(cb.emb, indexSpecLeft)

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
            .getOrAssert(cb)
            .asString
            .loadString(cb)
          val interval = ctxStruct
            .loadField(cb, "interval")
            .getOrAssert(cb)
            .asInterval
          index.initialize(cb, indexPath)

          val indexResult = index.queryInterval(cb, interval)
          val startIndex = indexResult.loadField(cb, 0)
            .getOrAssert(cb)
            .asInt64
            .value
          val endIndex = indexResult.loadField(cb, 1)
            .getOrAssert(cb)
            .asInt64
            .value
          cb.assign(curIdx, startIndex)
          cb.assign(endIdx, endIndex)

          cb.assign(
            partIdx,
            ctxStruct.loadField(cb, "partitionIndex").getOrAssert(cb).asInt64.value,
          )
          cb.assign(
            leftBuffer,
            specLeft.buildCodeInputBuffer(
              Code.newInstance[ByteTrackingInputStream, InputStream](
                mb.openUnbuffered(
                  ctxStruct.loadField(cb, "leftPartitionPath")
                    .getOrAssert(cb)
                    .asString
                    .loadString(cb),
                  true,
                )
              )
            ),
          )
          cb.assign(
            rightBuffer,
            specRight.buildCodeInputBuffer(
              Code.newInstance[ByteTrackingInputStream, InputStream](
                mb.openUnbuffered(
                  ctxStruct.loadField(cb, "rightPartitionPath")
                    .getOrAssert(cb)
                    .asString
                    .loadString(cb),
                  true,
                )
              )
            ),
          )

          cb.if_(
            endIndex > startIndex, {
              val leafNode = indexResult.loadField(cb, 2)
                .getOrAssert(cb)
                .asBaseStruct

              val leftSeekAddr = index.offsetAnnotation(cb, leafNode)
              cb += leftBuffer.seek(leftSeekAddr.asInt64.value)

              val rightSeekAddr = index.offsetAnnotation(cb, leafNode, altSpec = indexSpecRight)
              cb += rightBuffer.seek(rightSeekAddr.asInt64.value)
            },
          )

          index.close(cb)
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(curIdx >= endIdx, cb.goto(LendOfStream))
          val nextLeft = leftBuffer.readByte()
          cb.if_(nextLeft cne 1, cb._fatal(s"bad rows buffer state!"))
          val nextRight = rightBuffer.readByte()
          cb.if_(nextRight cne 1, cb._fatal(s"bad entries buffer state!"))
          cb.assign(curIdx, curIdx + 1L)
          cb.assign(leftValue, leftDec(cb, region, leftBuffer))
          cb.assign(rightValue, rightDec(cb, region, rightBuffer))
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb) { cb =>
          if (insertUID) {
            val uid = SStackStruct.constructFromArgs(
              cb,
              region,
              TTuple(TInt64, TInt64),
              EmitCode.present(mb, primitive(partIdx)),
              EmitCode.present(mb, primitive(cb.memoize(curIdx.get - 1L))),
            )
            val merged = SBaseStruct.merge(cb, leftValue.asBaseStruct, rightValue.asBaseStruct)
            IEmitCode.present(
              cb,
              merged._insert(requestedType, uidFieldName -> EmitValue.present(uid)),
            )
          } else {
            IEmitCode.present(
              cb,
              SBaseStruct.merge(cb, leftValue.asBaseStruct, rightValue.asBaseStruct),
            )
          }
        }

        override def close(cb: EmitCodeBuilder): Unit = {
          cb += leftBuffer.invoke[Unit]("close")
          cb += rightBuffer.invoke[Unit]("close")
        }
      }
      SStreamValue(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class TableNativeReaderParameters(
  path: String,
  options: Option[NativeReaderOptions],
)

class TableNativeReader(
  val params: TableNativeReaderParameters,
  val spec: AbstractTableSpec,
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = Array(params.path)

  val filterIntervals: Boolean = params.options.map(_.filterIntervals).getOrElse(false)

  def partitionCounts: Option[IndexedSeq[Long]] =
    if (params.options.isDefined) None else Some(spec.partitionCounts)

  override def isDistinctlyKeyed: Boolean = spec.isDistinctlyKeyed

  def uidType = TTuple(TInt64, TInt64)

  def fullTypeWithoutUIDs = spec.table_type

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(tcoerce[PStruct](spec.rowsComponent.rvdSpec(ctx.fs, params.path)
      .typedCodecSpec.encodedType.decodedPType(requestedType.rowType)))

  protected def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(tcoerce[PStruct](spec.globalsComponent.rvdSpec(ctx.fs, params.path)
      .typedCodecSpec.encodedType.decodedPType(requestedType.globalType)))

  override def toJValue: JValue = {
    implicit val formats: Formats = DefaultFormats
    decomposeWithName(params, "TableNativeReader")
  }

  override def renderShort(): String =
    s"(TableNativeReader ${params.path} ${params.options.map(_.renderShort()).getOrElse("")})"

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: TableNativeReader => params == that.params
    case _ => false
  }

  override def toString(): String = s"TableNativeReader($params)"

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globalsSpec = spec.globalsSpec
    val globalsPath = spec.globalsComponent.absolutePath(params.path)
    assert(!requestedGlobalsType.hasField(uidFieldName))
    ArrayRef(
      ToArray(ReadPartition(
        MakeStruct(Array(
          "partitionIndex" -> I64(0),
          "partitionPath" -> Str(globalsSpec.absolutePartPaths(globalsPath).head),
        )),
        requestedGlobalsType,
        PartitionNativeReader(globalsSpec.typedCodecSpec, uidFieldName),
      )),
      0,
    )
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globals = lowerGlobals(ctx, requestedType.globalType)
    val rowsSpec = spec.rowsSpec
    val specPart = rowsSpec.partitioner(ctx.stateManager)
    val partitioner = if (filterIntervals)
      params.options.map(opts =>
        RVDPartitioner.union(
          ctx.stateManager,
          specPart.kType,
          opts.intervals,
          specPart.kType.size - 1,
        )
      )
    else
      params.options.map(opts =>
        new RVDPartitioner(ctx.stateManager, specPart.kType, opts.intervals)
      )

    // If the data on disk already has a uidFieldName field, we should read it
    // as is. Do this by passing a dummy uidFieldName to the rows component,
    // which is not in the requestedType, so is ignored.
    val requestedUIDFieldName = if (spec.table_type.rowType.hasField(uidFieldName))
      "__dummy_uid"
    else
      uidFieldName

    spec.rowsSpec.readTableStage(
      ctx,
      spec.rowsComponent.absolutePath(params.path),
      requestedType,
      requestedUIDFieldName,
      partitioner,
      filterIntervals,
    ).apply(globals)
  }
}

case class TableNativeZippedReader(
  pathLeft: String,
  pathRight: String,
  options: Option[NativeReaderOptions],
  specLeft: AbstractTableSpec,
  specRight: AbstractTableSpec,
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = FastSeq(pathLeft, pathRight)

  override def renderShort(): String =
    s"(TableNativeZippedReader $pathLeft $pathRight ${options.map(_.renderShort()).getOrElse("")})"

  private lazy val filterIntervals = options.exists(_.filterIntervals)

  private def intervals = options.map(_.intervals)

  require(
    (specLeft.table_type.rowType.fieldNames ++ specRight.table_type.rowType.fieldNames).areDistinct()
  )

  require(specRight.table_type.key.isEmpty)
  require(specLeft.partitionCounts sameElements specRight.partitionCounts)
  require(specLeft.version == specRight.version)

  def partitionCounts: Option[IndexedSeq[Long]] =
    if (intervals.isEmpty) Some(specLeft.partitionCounts) else None

  override def uidType = TTuple(TInt64, TInt64)

  override def fullTypeWithoutUIDs: TableType = specLeft.table_type.copy(
    rowType = specLeft.table_type.rowType ++ specRight.table_type.rowType
  )

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

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(fieldInserter(
      ctx,
      leftPType(ctx, leftRType(requestedType.rowType)),
      rightPType(ctx, rightRType(requestedType.rowType)),
    )._1)

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalTuple(true, PInt64Required, PInt64Required))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(specLeft.globalsComponent.rvdSpec(ctx.fs, pathLeft)
      .typedCodecSpec.encodedType.decodedPType(requestedType.globalType))

  def fieldInserter(ctx: ExecuteContext, pLeft: PStruct, pRight: PStruct)
    : (PStruct, (HailClassLoader, FS, HailTaskContext, Region) => AsmFunction3RegionLongLongLong) = {
    val leftRef = Ref(freshName(), pLeft.virtualType)
    val rightRef = Ref(freshName(), pRight.virtualType)
    val (Some(PTypeReferenceSingleCodeType(t: PStruct)), mk) =
      Compile[AsmFunction3RegionLongLongLong](
        ctx,
        FastSeq(
          leftRef.name -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pLeft)),
          rightRef.name -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pRight)),
        ),
        FastSeq(typeInfo[Region], LongInfo, LongInfo),
        LongInfo,
        InsertFields(leftRef, pRight.fieldNames.map(f => f -> GetField(rightRef, f))),
      )
    (t, mk)
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globalsSpec = specLeft.globalsSpec
    val globalsPath = specLeft.globalsComponent.absolutePath(pathLeft)
    ArrayRef(
      ToArray(ReadPartition(
        MakeStruct(Array(
          "partitionIndex" -> I64(0),
          "partitionPath" -> Str(globalsSpec.absolutePartPaths(globalsPath).head),
        )),
        requestedGlobalsType,
        PartitionNativeReader(globalsSpec.typedCodecSpec, uidFieldName),
      )),
      0,
    )
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globals = lowerGlobals(ctx, requestedType.globalType)
    val rowsSpec = specLeft.rowsSpec
    val specPart = rowsSpec.partitioner(ctx.stateManager)
    val partitioner = if (filterIntervals)
      options.map(opts =>
        RVDPartitioner.union(
          ctx.stateManager,
          specPart.kType,
          opts.intervals,
          specPart.kType.size - 1,
        )
      )
    else
      options.map(opts => new RVDPartitioner(ctx.stateManager, specPart.kType, opts.intervals))

    AbstractRVDSpec.readZippedLowered(
      ctx,
      specLeft.rowsSpec,
      specRight.rowsSpec,
      pathLeft + "/rows",
      pathRight + "/rows",
      partitioner,
      filterIntervals,
      requestedType.rowType,
      requestedType.key,
      uidFieldName,
    ).apply(globals)
  }

}

object TableFromBlockMatrixNativeReader {
  def apply(fs: FS, params: TableFromBlockMatrixNativeReaderParameters)
    : TableFromBlockMatrixNativeReader = {
    val metadata: BlockMatrixMetadata = BlockMatrix.readMetadata(fs, params.path)
    TableFromBlockMatrixNativeReader(params, metadata)

  }

  def apply(
    fs: FS,
    path: String,
    nPartitions: Option[Int] = None,
    maximumCacheMemoryInBytes: Option[Int] = None,
  ): TableFromBlockMatrixNativeReader =
    TableFromBlockMatrixNativeReader(
      fs,
      TableFromBlockMatrixNativeReaderParameters(path, nPartitions, maximumCacheMemoryInBytes),
    )

  def fromJValue(fs: FS, jv: JValue): TableFromBlockMatrixNativeReader = {
    implicit val formats: Formats = TableReader.formats
    val params = jv.extract[TableFromBlockMatrixNativeReaderParameters]
    TableFromBlockMatrixNativeReader(fs, params)
  }
}

case class TableFromBlockMatrixNativeReaderParameters(
  path: String,
  nPartitions: Option[Int],
  maximumCacheMemoryInBytes: Option[Int],
)

case class TableFromBlockMatrixNativeReader(
  params: TableFromBlockMatrixNativeReaderParameters,
  metadata: BlockMatrixMetadata,
) extends TableReaderWithExtraUID {
  def pathsUsed: Seq[String] = FastSeq(params.path)

  val getNumPartitions: Int = params.nPartitions.getOrElse(HailContext.backend.defaultParallelism)

  val partitionRanges = (0 until getNumPartitions).map { i =>
    val nRows = metadata.nRows
    val start = (i * nRows) / getNumPartitions
    val end = ((i + 1) * nRows) / getNumPartitions
    start until end
  }

  override def partitionCounts: Option[IndexedSeq[Long]] =
    Some(partitionRanges.map(r => r.end - r.start))

  override def uidType = TInt64

  override def fullTypeWithoutUIDs: TableType = TableType(
    TStruct("row_idx" -> TInt64, "entries" -> TArray(TFloat64)),
    Array("row_idx"),
    TStruct.empty,
  )

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.rowType).setRequired(true))

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(PInt64Required)

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType)
    : VirtualTypeWithReq =
    VirtualTypeWithReq(PCanonicalStruct.empty(required = true))

  override def toExecuteIntermediate(
    ctx: ExecuteContext,
    requestedType: TableType,
    dropRows: Boolean,
  ): TableExecuteIntermediate = {
    assert(!dropRows)
    val rowsRDD = new BlockMatrixReadRowBlockedRDD(
      ctx.fsBc,
      params.path,
      partitionRanges,
      requestedType.rowType,
      metadata,
      maybeMaximumCacheMemoryInBytes = params.maximumCacheMemoryInBytes,
    )

    val partitionBounds = partitionRanges.map { r =>
      Interval(Row(r.start), Row(r.end), true, false)
    }
    val partitioner = new RVDPartitioner(ctx.stateManager, fullType.keyType, partitionBounds)

    val rowTyp = PType.canonical(requestedType.rowType, required = true).asInstanceOf[PStruct]
    val rvd =
      RVD(RVDType(rowTyp, fullType.key.filter(rowTyp.hasField)), partitioner, ContextRDD(rowsRDD))
    TableExecuteIntermediate(TableValue(ctx, requestedType, BroadcastRow.empty(ctx), rvd))
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    throw new LowererUnsupportedOperation(s"${getClass.getSimpleName}.lower not implemented")

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    throw new LowererUnsupportedOperation(s"${getClass.getSimpleName}.lowerGlobals not implemented")

  override def toJValue: JValue =
    decomposeWithName(params, "TableFromBlockMatrixNativeReader")(TableReader.formats)

  def renderShort(): String = defaultRender()
}

object TableRead {
  def native(fs: FS, path: String, uidField: Boolean = false): TableRead = {
    val tr = TableNativeReader(fs, TableNativeReaderParameters(path, None))
    val requestedType = if (uidField)
      tr.fullType
    else
      tr.fullType.copy(
        rowType = tr.fullType.rowType.deleteKey(TableReader.uidFieldName)
      )
    TableRead(requestedType, false, tr)
  }
}

case class TableRead(typ: TableType, dropRows: Boolean, tr: TableReader) extends TableIR {
  try
    assert(PruneDeadFields.isSupertype(typ, tr.fullType))
  catch {
    case e: Throwable =>
      fatal(s"bad type:\n  full type: ${tr.fullType}\n  requested: $typ\n  reader: $tr", e)
  }

  def isDistinctlyKeyed: Boolean = tr.isDistinctlyKeyed

  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(typ, dropRows, tr)
  }
}

case class TableParallelize(rowsAndGlobal: IR, nPartitions: Option[Int] = None) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = FastSeq(rowsAndGlobal)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableParallelize = {
    val IndexedSeq(newrowsAndGlobal: IR) = newChildren
    TableParallelize(newrowsAndGlobal, nPartitions)
  }

  lazy val typ: TableType = {
    def rowsType = rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("rows").asInstanceOf[TArray]
    def globalsType =
      rowsAndGlobal.typ.asInstanceOf[TStruct].fieldType("global").asInstanceOf[TStruct]
    TableType(
      rowsType.elementType.asInstanceOf[TStruct],
      FastSeq(),
      globalsType,
    )
  }
}

/** Change the table to have key 'keys'.
  *
  * Let n be the longest common prefix of 'keys' and the old key, i.e. the number of key fields that
  * are not being changed.
  *   - If 'isSorted', then 'child' must already be sorted by 'keys', and n must not be zero. Thus,
  *     if 'isSorted', TableKeyBy will not shuffle or scan. The new partitioner will be the old one
  *     with partition bounds truncated to length n.
  *   - If n = 'keys.length', i.e. we are simply shortening the key, do nothing but change the table
  *     type to the new key. 'isSorted' is ignored.
  *   - Otherwise, if 'isSorted' is false and n < 'keys.length', then shuffle.
  */
case class TableKeyBy(child: TableIR, keys: IndexedSeq[String], isSorted: Boolean = false)
    extends TableIR with PreservesRows {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  lazy val typ: TableType = child.typ.copy(key = keys)

  def definitelyDoesNotShuffle: Boolean = child.typ.key.startsWith(keys) || isSorted

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, isSorted)
  }

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

/** Generate a table from the elementwise application of a body IR to a stream of `contexts`.
  *
  * @param contexts
  *   IR of type TStream[Any] whose elements are downwardly exposed to `body` as `cname`.
  * @param globals
  *   IR of type TStruct, downwardly exposed to `body` as `gname`.
  * @param cname
  *   Name of free variable in `body` referencing elements of `contexts`.
  * @param gname
  *   Name of free variable in `body` referencing `globals`.
  * @param body
  *   IR of type TStream[TStruct] that generates the rows of the table for each element in
  *   `contexts`, optionally referencing free variables Ref(cname) and Ref(gname).
  * @param partitioner
  * @param errorId
  *   Identifier tracing location in Python source that created this node
  */
case class TableGen(
  contexts: IR,
  globals: IR,
  cname: Name,
  gname: Name,
  body: IR,
  partitioner: RVDPartitioner,
  errorId: Int = ErrorIDs.NO_ERROR,
) extends TableIR {
  private def globalType =
    TypeCheck.coerce[TStruct]("globals", globals.typ)

  private def rowType = {
    val bodyType = TypeCheck.coerce[TStream]("body", body.typ)
    TypeCheck.coerce[TStruct]("body.elementType", bodyType.elementType)
  }

  override lazy val typ: TableType =
    TableType(rowType, partitioner.kType.fieldNames, globalType)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(contexts: IR, globals: IR, body: IR) = newChildren
    TableGen(contexts, globals, cname, gname, body, partitioner, errorId)
  }

  override def childrenSeq: IndexedSeq[BaseIR] =
    FastSeq(contexts, globals, body)
}

case class TableRange(n: Int, nPartitions: Int) extends TableIR {
  require(n >= 0)
  require(nPartitions > 0)
  private val nPartitionsAdj = math.max(math.min(n, nPartitions), 1)
  val childrenSeq: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableRange = {
    assert(newChildren.isEmpty)
    TableRange(n, nPartitions)
  }

  val partitionCounts: IndexedSeq[Int] = partition(n, nPartitionsAdj).toFastSeq

  val typ: TableType = TableType(
    TStruct("idx" -> TInt32),
    Array("idx"),
    TStruct.empty,
  )
}

case class TableFilter(child: TableIR, pred: IR) extends TableIR with PreservesOrRemovesRows {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, pred)

  def typ: TableType = child.typ

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableFilter = {
    assert(newChildren.length == 2)
    TableFilter(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableHead(child: TableIR, n: Long) extends TableIR {
  require(n >= 0, fatal(s"TableHead: n must be non-negative! Found '$n'."))
  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)
  def typ: TableType = child.typ

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableHead = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableHead(newChild, n)
  }
}

case class TableTail(child: TableIR, n: Long) extends TableIR {
  require(n >= 0, fatal(s"TableTail: n must be non-negative! Found '$n'."))
  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)
  def typ: TableType = child.typ

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableTail = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableTail(newChild, n)
  }
}

object RepartitionStrategy {
  val SHUFFLE: Int = 0
  val COALESCE: Int = 1
  val NAIVE_COALESCE: Int = 2
}

case class TableRepartition(child: TableIR, n: Int, strategy: Int)
    extends TableIR with PreservesRows {
  def typ: TableType = child.typ

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableRepartition = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRepartition(newChild, n, strategy)
  }

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

object TableJoin {
  def apply(left: TableIR, right: TableIR, joinType: String): TableJoin =
    TableJoin(left, right, joinType, left.typ.key.length)
}

/** Suppose 'left' has key [l_1, ..., l_n] and 'right' has key [r_1, ..., r_m]. Then [l_1, ..., l_j]
  * and [r_1, ..., r_j] must have the same type, where j = 'joinKey'. TableJoin computes the join of
  * 'left' and 'right' along this common prefix of their keys, returning a table with key [l_1, ...,
  * l_j, l_{j+1}, ..., l_n, r_{j+1}, ..., r_m].
  *
  * WARNING: If 'left' has any duplicate (full) key [k_1, ..., k_n], and j < m, and 'right' has
  * multiple rows with the corresponding join key [k_1, ..., k_j] but distinct full keys, then the
  * resulting table will have out-of-order keys. To avoid this, ensure one of the following: * j ==
  * m * 'left' has distinct keys * 'right' has distinct join keys (length j prefix), or at least no
  * distinct keys with the same join key.
  */
case class TableJoin(left: TableIR, right: TableIR, joinType: String, joinKey: Int)
    extends TableIR {

  require(joinKey >= 0)

  require(joinType == "inner" ||
    joinType == "left" ||
    joinType == "right" ||
    joinType == "outer")

  val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  lazy val typ: TableType = {
    val leftRowType = left.typ.rowType
    val rightRowType = right.typ.rowType
    val leftKey = left.typ.key.take(joinKey)
    val rightKey = right.typ.key.take(joinKey)

    val leftKeyType = TableType.keyType(leftRowType, leftKey)
    val leftValueType = TableType.valueType(leftRowType, leftKey)
    val rightValueType = TableType.valueType(rightRowType, rightKey)
    if (
      leftValueType.fieldNames.toSet
        .intersect(rightValueType.fieldNames.toSet)
        .nonEmpty
    )
      throw new RuntimeException(
        s"invalid join: \n  left value:  $leftValueType\n  right value: $rightValueType"
      )

    val newRowType = leftKeyType ++ leftValueType ++ rightValueType
    val newGlobalType = left.typ.globalType ++ right.typ.globalType
    val newKey = left.typ.key ++ right.typ.key.drop(joinKey)

    TableType(newRowType, newKey, newGlobalType)
  }

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableJoin = {
    assert(newChildren.length == 2)
    TableJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      joinType,
      joinKey,
    )
  }
}

case class TableIntervalJoin(
  left: TableIR,
  right: TableIR,
  root: String,
  product: Boolean,
) extends TableIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  lazy val typ: TableType = {
    val rightType: Type = if (product) TArray(right.typ.valueType) else right.typ.valueType
    left.typ.copy(rowType = left.typ.rowType.appendKey(root, rightType))
  }

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR =
    TableIntervalJoin(
      newChildren(0).asInstanceOf[TableIR],
      newChildren(1).asInstanceOf[TableIR],
      root,
      product,
    )

  override def preservesRowsOrColsFrom: BaseIR = left
}

/** The TableMultiWayZipJoin node assumes that input tables have distinct keys. If inputs do not
  * have distinct keys, the key that is included in the result is undefined, but is likely the last.
  */
case class TableMultiWayZipJoin(
  childrenSeq: IndexedSeq[TableIR],
  fieldName: String,
  globalName: String,
) extends TableIR {
  require(childrenSeq.nonEmpty, "there must be at least one table as an argument")

  private def first = childrenSeq.head

  lazy val typ: TableType = {
    def newGlobalType = TStruct(globalName -> TArray(first.typ.globalType))
    def newValueType = TStruct(fieldName -> TArray(first.typ.valueType))
    def newRowType = first.typ.keyType ++ newValueType
    first.typ.copy(
      rowType = newRowType,
      globalType = newGlobalType,
    )
  }

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : TableMultiWayZipJoin =
    TableMultiWayZipJoin(newChildren.asInstanceOf[IndexedSeq[TableIR]], fieldName, globalName)
}

case class TableLeftJoinRightDistinct(left: TableIR, right: TableIR, root: String)
    extends TableIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(left, right)

  lazy val typ: TableType = left.typ.copy(
    rowType = left.typ.rowType.structInsert(right.typ.valueType, FastSeq(root))
  )

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : TableLeftJoinRightDistinct = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableLeftJoinRightDistinct(newLeft, newRight, root)
  }

  override def preservesRowsOrColsFrom: BaseIR = left
}

object TableMapPartitions {
  def apply(child: TableIR, globalName: Name, partitionStreamName: Name, body: IR)
    : TableMapPartitions =
    TableMapPartitions(child, globalName, partitionStreamName, body, 0, child.typ.key.length)
}

case class TableMapPartitions(
  child: TableIR,
  globalName: Name,
  partitionStreamName: Name,
  body: IR,
  requestedKey: Int,
  allowedOverlap: Int,
) extends TableIR {
  lazy val typ: TableType = child.typ.copy(
    rowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct]
  )

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, body)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : TableMapPartitions = {
    assert(newChildren.length == 2)
    TableMapPartitions(
      newChildren(0).asInstanceOf[TableIR],
      globalName,
      partitionStreamName,
      newChildren(1).asInstanceOf[IR],
      requestedKey,
      allowedOverlap,
    )
  }
}

// Must leave key fields unchanged.
case class TableMapRows(child: TableIR, newRow: IR) extends TableIR with PreservesRows {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, newRow)

  lazy val typ: TableType = child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableMapGlobals(child: TableIR, newGlobals: IR) extends TableIR with PreservesRows {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child, newGlobals)

  lazy val typ: TableType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableExplode(child: TableIR, path: IndexedSeq[String]) extends TableIR {
  assert(path.nonEmpty)

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  private def childRowType = child.typ.rowType

  lazy val typ: TableType = child.typ.copy(rowType =
    childRowType.structInsert(
      tcoerce[TContainer](childRowType.queryTyped(path)._1).elementType,
      path,
    )
  )

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableExplode = {
    assert(newChildren.length == 1)
    TableExplode(newChildren(0).asInstanceOf[TableIR], path)
  }
}

case class TableUnion(childrenSeq: IndexedSeq[TableIR]) extends TableIR {
  assert(childrenSeq.nonEmpty)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableUnion =
    TableUnion(newChildren.map(_.asInstanceOf[TableIR]))

  def typ: TableType = childrenSeq(0).typ
}

case class MatrixRowsTable(child: MatrixIR) extends TableIR with PreservesRows {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixRowsTable = {
    assert(newChildren.length == 1)
    MatrixRowsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  def typ: TableType = child.typ.rowsTableType

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  def typ: TableType = child.typ.colsTableType
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  def typ: TableType = child.typ.entriesTableType
}

case class TableDistinct(child: TableIR) extends TableIR with PreservesOrRemovesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableDistinct = {
    val IndexedSeq(newChild) = newChildren
    TableDistinct(newChild.asInstanceOf[TableIR])
  }

  def typ: TableType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableKeyByAndAggregate(
  child: TableIR,
  expr: IR,
  newKey: IR,
  nPartitions: Option[Int] = None,
  bufferSize: Int,
) extends TableIR with PreservesOrRemovesRows {
  assert(bufferSize > 0)

  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, expr, newKey)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : TableKeyByAndAggregate = {
    val IndexedSeq(newChild: TableIR, newExpr: IR, newNewKey: IR) = newChildren
    TableKeyByAndAggregate(newChild, newExpr, newNewKey, nPartitions, bufferSize)
  }

  private lazy val keyType = newKey.typ.asInstanceOf[TStruct]

  lazy val typ: TableType = TableType(
    rowType = keyType ++ tcoerce[TStruct](expr.typ),
    globalType = child.typ.globalType,
    key = keyType.fieldNames,
  )

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

// follows key_by non-empty key
case class TableAggregateByKey(child: TableIR, expr: IR)
    extends TableIR with PreservesOrRemovesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child, expr)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR])
    : TableAggregateByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: TableIR, newExpr: IR) = newChildren
    TableAggregateByKey(newChild, newExpr)
  }

  lazy val typ: TableType =
    child.typ.copy(rowType = child.typ.keyType ++ tcoerce[TStruct](expr.typ))

  override def preservesRowsOrColsFrom: BaseIR = child
}

object TableOrderBy {
  def isAlreadyOrdered(sortFields: IndexedSeq[SortField], prevKey: IndexedSeq[String]): Boolean =
    sortFields.length <= prevKey.length &&
      sortFields.zip(prevKey).forall { case (sf, k) =>
        sf.sortOrder == Ascending && sf.field == k
      }
}

case class TableOrderBy(child: TableIR, sortFields: IndexedSeq[SortField])
    extends TableIR with PreservesRows {
  lazy val definitelyDoesNotShuffle: Boolean =
    TableOrderBy.isAlreadyOrdered(sortFields, child.typ.key)
  // TableOrderBy expects an unkeyed child, so that we can better optimize by
  // pushing these two steps around as needed

  val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableOrderBy = {
    val IndexedSeq(newChild) = newChildren
    TableOrderBy(newChild.asInstanceOf[TableIR], sortFields)
  }

  lazy val typ: TableType = child.typ.copy(key = FastSeq())

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesPartitioning: Boolean = false
}

/** Create a Table from a MatrixTable, storing the column values in a global field 'colsFieldName',
  * and storing the entry values in a row field 'entriesFieldName'.
  */
case class CastMatrixToTable(
  child: MatrixIR,
  entriesFieldName: String,
  colsFieldName: String,
) extends TableIR with PreservesRows {

  lazy val typ: TableType = child.typ.toTableType(entriesFieldName, colsFieldName)

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): CastMatrixToTable = {
    val IndexedSeq(newChild) = newChildren
    CastMatrixToTable(newChild.asInstanceOf[MatrixIR], entriesFieldName, colsFieldName)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableRename(child: TableIR, rowMap: Map[String, String], globalMap: Map[String, String])
    extends TableIR with PreservesRows {
  def rowF(old: String): String = rowMap.getOrElse(old, old)

  lazy val typ: TableType = child.typ.copy(
    rowType = child.typ.rowType.rename(rowMap),
    globalType = child.typ.globalType.rename(globalMap),
    key = child.typ.key.map(k => rowMap.getOrElse(k, k)),
  )

  lazy val childrenSeq: IndexedSeq[BaseIR] = FastSeq(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableRename = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRename(newChild, rowMap, globalMap)
  }

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class TableFilterIntervals(child: TableIR, intervals: IndexedSeq[Interval], keep: Boolean)
    extends TableIR with PreservesOrRemovesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableFilterIntervals(newChild, intervals, keep)
  }

  override def typ: TableType = child.typ

  override def preservesRowsOrColsFrom: BaseIR = child
}

case class MatrixToTableApply(child: MatrixIR, function: MatrixToTableFunction)
    extends TableIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: MatrixIR) = newChildren
    MatrixToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def preservesRowsOrColsFrom: BaseIR = child
  override def preservesRowsCond: Boolean = function.preservesPartitionCounts
}

case class TableToTableApply(child: TableIR, function: TableToTableFunction)
    extends TableIR with PreservesRows {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def preservesRowsOrColsFrom: BaseIR = child

  override def preservesRowsCond: Boolean = function.preservesPartitionCounts
}

case class BlockMatrixToTableApply(
  bm: BlockMatrixIR,
  aux: IR,
  function: BlockMatrixToTableFunction,
) extends TableIR {

  override lazy val childrenSeq: IndexedSeq[BaseIR] = Array(bm, aux)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR =
    BlockMatrixToTableApply(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[IR],
      function,
    )

  override lazy val typ: TableType = function.typ(bm.typ, aux.typ)
}

case class BlockMatrixToTable(child: BlockMatrixIR) extends TableIR {
  lazy val childrenSeq: IndexedSeq[BaseIR] = Array(child)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixToTable(newChild)
  }

  override val typ: TableType = {
    val rvType = TStruct("i" -> TInt64, "j" -> TInt64, "entry" -> TFloat64)
    TableType(rvType, Array[String](), TStruct.empty)
  }
}

case class RelationalLetTable(name: Name, value: IR, body: TableIR)
    extends TableIR with PreservesRows {
  def typ: TableType = body.typ

  def childrenSeq: IndexedSeq[BaseIR] = Array(value, body)

  override protected def copyWithNewChildren(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newValue: IR, newBody: TableIR) = newChildren
    RelationalLetTable(name, newValue, newBody)
  }

  def preservesRowsOrColsFrom: BaseIR = body
}
