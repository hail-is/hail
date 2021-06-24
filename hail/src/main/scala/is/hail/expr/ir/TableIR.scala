package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.HailTaskContext
import is.hail.backend.spark.{SparkBackend, SparkTaskContext}
import is.hail.expr.ir
import is.hail.expr.ir.functions.{BlockMatrixToTableFunction, MatrixToTableFunction, TableToTableFunction}
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage, TableStageDependency}
import is.hail.expr.ir.streams.{StreamArgType, StreamProducer}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.index.{IndexReadIterator, IndexReader, IndexReaderBuilder, MaybeIndexedReadZippedIterator}
import is.hail.linalg.{BlockMatrix, BlockMatrixMetadata, BlockMatrixReadRowBlockedRDD}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.types._
import is.hail.types.physical.{stypes, _}
import is.hail.types.physical.stypes.{BooleanSingleCodeType, Int32SingleCodeType, PTypeReferenceSingleCodeType, StreamSingleCodeType}
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStream, SStreamCode}
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.TaskContext
import org.apache.spark.executor.InputMetrics
import org.apache.spark.sql.Row
import org.json4s.JsonAST.JString
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Extraction, Formats, JValue, ShortTypeHints}

import java.io.{ByteArrayInputStream, DataInputStream, DataOutputStream, InputStream}
import scala.reflect.ClassTag

object TableIR {
  def read(fs: FS, path: String, dropRows: Boolean = false, requestedType: Option[TableType] = None): TableIR = {
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

  protected[ir] def execute(ctx: ExecuteContext): TableValue =
    fatal("tried to execute unexecutable IR:\n" + Pretty(this))

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
  def apply(value: TableValue): TableLiteral = {
    val globalPType = PType.canonical(value.globals.t)
    val enc = TypedCodecSpec(globalPType, BufferSpec.wireSpec) // use wireSpec to save memory
    using(new ByteArrayEncoder(enc.buildEncoder(value.ctx, value.globals.t))) { encoder =>
      TableLiteral(value.typ, value.rvd, enc,
        encoder.regionValueToBytes(value.globals.value.offset))
    }
  }
}

case class TableLiteral(typ: TableType, rvd: RVD, enc: AbstractTypedCodecSpec, encodedGlobals: Array[Byte]) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): TableLiteral = {
    assert(newChildren.isEmpty)
    TableLiteral(typ, rvd, enc, encodedGlobals)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val (globalPType: PStruct, dec) = enc.buildDecoder(ctx, typ.globalType)

    val bais = new ByteArrayInputStream(encodedGlobals)
    val globalOffset = dec.apply(bais).readRegionValue(ctx.r)
    TableValue(ctx, typ, BroadcastRow(ctx, RegionValue(ctx.r, globalOffset), globalPType), rvd)
  }
}

object TableReader {
  implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(
    List(classOf[TableNativeZippedReader])
  ) + new NativeReaderOptionsSerializer()

  def fromJValue(fs: FS, jv: JValue): TableReader = {
    (jv \ "name").extract[String] match {
      case "TableNativeReader" => TableNativeReader.fromJValue(fs, jv)
      case "TextTableReader" => TextTableReader.fromJValue(fs, jv)
      case "TableFromBlockMatrixNativeReader" => TableFromBlockMatrixNativeReader.fromJValue(fs, jv)
      case _ => jv.extract[TableReader]
    }
  }
}

object LoweredTableReader {
  def makeCoercer(
    ctx: ExecuteContext,
    key: IndexedSeq[String],
    partitionKey: Int,
    contextType: Type,
    contexts: IndexedSeq[Any],
    keyType: TStruct,
    keyPType: (TStruct) => PStruct,
    keys: (TStruct) => (Region, Any) => Iterator[Long]
  ): LoweredTableReaderCoercer = {
    assert(key.nonEmpty)
    assert(contexts.nonEmpty)

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
          contextType,
          (requestedType: Type) => keyPType(requestedType.asInstanceOf[TStruct]),
          (requestedType: Type) => keys(requestedType.asInstanceOf[TStruct]))),
        "key",
        MakeStruct(FastIndexedSeq(
          "key" -> Ref("key", keyType),
          "token" -> invokeSeeded("rand_unif", 1, TFloat64, F64(0.0), F64(1.0)),
          "prevkey" -> ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(Ref("key", keyType)), prevkey)))),
      "x",
      Let("n", ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(), count),
        AggLet("key", GetField(Ref("x", xType), "key"),
          MakeStruct(FastIndexedSeq(
            "n" -> Ref("n", TInt64),
            "minkey" ->
              ArrayRef(
                ApplyAggOp(
                  FastIndexedSeq(I32(1)),
                  FastIndexedSeq(Ref("key", keyType), Ref("key", keyType)),
                  minkey),
                I32(0)),
            "maxkey" ->
              ArrayRef(
                ApplyAggOp(
                  FastIndexedSeq(I32(1)),
                  FastIndexedSeq(Ref("key", keyType), Ref("key", keyType)),
                  maxkey),
                I32(0)),
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
      scanBody(Ref("context", contextType)))

    val partDataWithIndex = InsertFields(
      ArrayRef(Ref("scanResult", scanResult.typ), Ref("i", TInt32)),
      FastIndexedSeq(
        "i" -> Ref("i", TInt32)))
    val sortedPartDataIR = ArraySort(
      Let("scanResult", scanResult,
        StreamMap(
          StreamRange(I32(0), ArrayLen(Ref("scanResult", scanResult.typ)), I32(1)),
          "i",
          partDataWithIndex)),
      "l", "r",
      ApplyComparisonOp(LT(keyType),
        GetField(Ref("l", partDataWithIndex.typ), "minkey"),
        GetField(Ref("r", partDataWithIndex.typ), "minkey")))

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
                  GetField(Ref("partDataWithIndex", partDataWithIndex.typ), "ksorted"))),
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
                  GetField(Ref("partDataWithIndex", partDataWithIndex.typ), "pksorted"))),
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

    val (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) = Compile[AsmFunction1RegionLong](ctx,
      FastIndexedSeq(),
      FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
      summary,
      optimize = true)

    val a = f(ctx.fs, 0, ctx.r)(ctx.r)
    val s = SafeRow(resultPType, a)

    val ksorted = s.getBoolean(0)
    val pksorted = s.getBoolean(1)
    val sortedPartData = s.getAs[IndexedSeq[Row]](2)

    if (ksorted) {
      info("Coerced sorted dataset")

      new LoweredTableReaderCoercer {
        def coerce(globals: IR,
          contextType: Type,
          contexts: IndexedSeq[Any],
          body: IR => IR): TableStage = {
          val partOrigIndex = sortedPartData.map(_.getInt(6))

          val partitioner = new RVDPartitioner(keyType,
            sortedPartData.map { partData =>
              Interval(partData.get(1), partData.get(2), includesStart = true, includesEnd = true)
            },
            key.length)

          TableStage(globals, partitioner, TableStageDependency.none,
            ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
            body)
        }
      }
    } else if (pksorted) {
      info("Coerced prefix-sorted dataset")

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

        def coerce(globals: IR,
          contextType: Type,
          contexts: IndexedSeq[Any],
          body: IR => IR): TableStage = {
          val partOrigIndex = sortedPartData.map(_.getInt(6))

          val partitioner = new RVDPartitioner(pkType,
            sortedPartData.map { partData =>
              Interval(selectPK(partData.getAs[Row](1)), selectPK(partData.getAs[Row](2)), includesStart = true, includesEnd = true)
            }, pkType.size)

          val pkPartitioned = TableStage(globals, partitioner, TableStageDependency.none,
            ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
            body)

          pkPartitioned
            .strictify()
            .mapPartition(None) { part =>
              flatMapIR(StreamGroupByKey(part, pkType.fieldNames)) { inner =>
                ToStream(sortIR(inner) { case (l, r) => ApplyComparisonOp(LT(l.typ), l, r) })
              }
            }.changePartitionerNoRepartition(partitioner.extendKeySamePartitions(keyType))
        }
      }
    } else {
      info(s"Ordering unsorted dataset with shuffle")

      new LoweredTableReaderCoercer {
        def coerce(globals: IR,
          contextType: Type,
          contexts: IndexedSeq[Any],
          body: IR => IR): TableStage = {
          val partOrigIndex = sortedPartData.map(_.getInt(6))

          val partitioner = RVDPartitioner.unkeyed(sortedPartData.length)

          val tableStage = TableStage(globals, partitioner, TableStageDependency.none,
            ToStream(Literal(TArray(contextType), partOrigIndex.map(i => contexts(i)))),
            body)

          val rowRType = TypeWithRequiredness(tableStage.rowType).asInstanceOf[RStruct]

          ctx.backend.lowerDistributedSort(ctx,
            tableStage,
            keyType.fieldNames.map(f => SortField(f, Ascending)),
            Map.empty,
            rowRType
          )
        }
      }
    }
  }
}

abstract class TableReader {
  def pathsUsed: Seq[String]

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue

  def partitionCounts: Option[IndexedSeq[Long]]

  def fullType: TableType

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct)

  def toJValue: JValue = {
    Extraction.decompose(this)(TableReader.formats)
  }

  def renderShort(): String

  def defaultRender(): String = {
    StringEscapeUtils.escapeString(JsonMethods.compact(toJValue))
  }

  def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    throw new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lowerGlobals not implemented")

  def lower(ctx: ExecuteContext, requestedType: TableType): TableStage =
    throw new LowererUnsupportedOperation(s"${ getClass.getSimpleName }.lower not implemented")
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


case class PartitionRVDReader(rvd: RVD) extends PartitionReader {
  override def contextType: Type = TInt32

  override def rowPType(requestedType: Type): PType = rvd.rowPType.subsetTo(requestedType)

  override def fullRowType: Type = rvd.rowType

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    partitionRegion: Value[Region],
    requestedType: Type): IEmitCode = {

    val mb = cb.emb

    val (Some(PTypeReferenceSingleCodeType(upcastPType)), upcast) = Compile[AsmFunction2RegionLongLong](ctx,
      FastIndexedSeq(("elt", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo),
      LongInfo,
      PruneDeadFields.upcast(Ref("elt", rvd.rowType), requestedType))

    val upcastCode = mb.getObject[Function3[FS, Int, Region, AsmFunction2RegionLongLong]](upcast)

    assert(upcastPType == rowPType(requestedType),
      s"ptype mismatch:\n  upcast: $upcastPType\n  computed: ${ rowPType(requestedType) }")

    context.toI(cb).map(cb) { idx =>
      val iterator = mb.genFieldThisRef[Iterator[Long]]("rvdreader_iterator")
      val next = mb.genFieldThisRef[Long]("rvdreader_next")

      val region = mb.genFieldThisRef[Region]("rvdreader_region")
      val upcastF = mb.genFieldThisRef[AsmFunction2RegionLongLong]("rvdreader_upcast")

      val broadcastRVD = mb.getObject[BroadcastRVD](new BroadcastRVD(ctx.backend.asSpark("RVDReader"), rvd))

      val producer = new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          cb.assign(iterator, broadcastRVD.invoke[Int, Region, Region, Iterator[Long]](
            "computePartition", EmitCodeBuilder.scopedCode[Int](mb)(idx.asInt.intCode(_)), region, partitionRegion))
          cb.assign(upcastF, Code.checkcast[AsmFunction2RegionLongLong](upcastCode.invoke[AnyRef, AnyRef, AnyRef, AnyRef]("apply", cb.emb.ecb.emodb.getFS, Code.boxInt(0), partitionRegion)))
        }
        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!iterator.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(next, upcastF.invoke[Region, Long, Long]("apply", region, Code.longValue(iterator.invoke[java.lang.Long]("next"))))
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, upcastPType.loadCheapPCode(cb, next)))

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamCode(producer)
    }
  }

  def toJValue: JValue = JString("<PartitionRVDReader>") // cannot be parsed, but need a printout for Pretty
}

trait AbstractNativeReader extends PartitionReader {
  def spec: AbstractTypedCodecSpec

  def rowPType(requestedType: Type): PType = spec.decodedPType(requestedType)

  def fullRowType: Type = spec.encodedVirtualType
}

case class PartitionNativeReader(spec: AbstractTypedCodecSpec) extends AbstractNativeReader {
  def contextType: Type = TString

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    partitionRegion: Value[Region],
    requestedType: Type): IEmitCode = {

    val mb = cb.emb

    context.toI(cb).map(cb) { path =>
      val pathString = path.asString.loadString()
      val xRowBuf = mb.genFieldThisRef[InputBuffer]("pnr_xrowbuf")
      val next = mb.newPSettable(mb.fieldBuilder, spec.encodedType.decodedSType(requestedType), "pnr_next")
      val region = mb.genFieldThisRef[Region]("pnr_region")

      val producer = new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          cb.assign(xRowBuf, spec.buildCodeInputBuffer(mb.open(pathString, checkCodec = true)))
        }
        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!xRowBuf.readByte().toZ, cb.goto(LendOfStream))
          cb.assign(next, spec.encodedType.buildDecoder(requestedType, cb.emb.ecb).apply(cb, region, xRowBuf))
          cb.goto(LproduceElementDone)
        }

        override val element: EmitCode = EmitCode.present(mb, next)

        override def close(cb: EmitCodeBuilder): Unit = cb += xRowBuf.close()
      }
      SStreamCode(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class PartitionNativeReaderIndexed(spec: AbstractTypedCodecSpec, indexSpec: AbstractIndexSpec, key: IndexedSeq[String]) extends AbstractNativeReader {
  def contextType: Type = TStruct(
    "partitionPath" -> TString,
    "indexPath" -> TString,
    "interval" -> RVDPartitioner.intervalIRRepresentation(spec.encodedVirtualType.asInstanceOf[TStruct].select(key)._1))

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    partitionRegion: Value[Region],
    requestedType: Type): IEmitCode = {

    val mb = cb.emb

    val (eltType, makeDec) = spec.buildDecoder(ctx, requestedType)

    val (keyType, annotationType) = indexSpec.types
    val (leafPType: PStruct, leafDec) = indexSpec.leafCodec.buildDecoder(ctx, indexSpec.leafCodec.encodedVirtualType)
    val (intPType: PStruct, intDec) = indexSpec.internalNodeCodec.buildDecoder(ctx, indexSpec.internalNodeCodec.encodedVirtualType)
    val mkIndexReader = IndexReaderBuilder.withDecoders(leafDec, intDec, keyType, annotationType, leafPType, intPType)

    val makeIndexCode = mb.getObject[Function4[FS, String, Int, RegionPool, IndexReader]](mkIndexReader)
    val makeDecCode = mb.getObject[(InputStream => Decoder)](makeDec)

    context.toI(cb).map(cb) { ctxStruct =>

      val getIndexReader: Code[String] => Code[IndexReader] = { (indexPath: Code[String]) =>
        Code.checkcast[IndexReader](
          makeIndexCode.invoke[AnyRef, AnyRef, AnyRef, AnyRef, AnyRef]("apply", mb.getFS, indexPath, Code.boxInt(8), mb.ecb.pool()))
      }

      val next = mb.newLocal[Long]("pnr_next")
      val idxr = mb.genFieldThisRef[IndexReader]("pnri_idx_reader")
      val it = mb.genFieldThisRef[IndexReadIterator]("pnri_idx_iterator")

      val region = mb.genFieldThisRef[Region]("pnr_region")

      val producer = new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          val ctxMemo = ctxStruct.asBaseStruct.memoize(cb, "pnri_ctx_struct")
          cb.assign(idxr, getIndexReader(ctxMemo
            .loadField(cb, "indexPath")
            .get(cb)
            .asString
            .loadString()))
          cb.assign(it,
            Code.newInstance7[IndexReadIterator,
              (InputStream) => Decoder,
              Region,
              InputStream,
              IndexReader,
              String,
              Interval,
              InputMetrics](makeDecCode,
              region,
              mb.open(ctxMemo.loadField(cb, "partitionPath")
                .get(cb)
                .asString
                .loadString(), true),
              idxr,
              Code._null[String],
              ctxMemo.loadField(cb, "interval")
                .consumeCode[Interval](cb,
                  Code._fatal[Interval](""),
                  { pc =>
                    val pcm = pc.memoize(cb, "pnri_interval")
                    val pt = pcm.st.canonicalPType()
                    Code.invokeScalaObject2[PType, Long, Interval](
                      PartitionBoundOrdering.getClass,
                      "regionValueToJavaObject",
                      mb.getPType(pt),
                      pt.store(cb, region, pcm, false))
                  }
                ),
              Code._null[InputMetrics]
            ))
        }
        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(next, it.invoke[Long]("_next"))
          cb.goto(LproduceElementDone)

        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, eltType.loadCheapPCode(cb, next)))

        override def close(cb: EmitCodeBuilder): Unit = cb += it.invoke[Unit]("close")
      }
      SStreamCode(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

case class PartitionZippedNativeReader(specLeft: AbstractTypedCodecSpec, specRight: AbstractTypedCodecSpec,
  indexSpecLeft: Option[AbstractIndexSpec], indexSpecRight: Option[AbstractIndexSpec],
  key: IndexedSeq[String]) extends PartitionReader {

  require(!(indexSpecLeft.isEmpty ^ indexSpecRight.isEmpty))

  def contextType: Type = {
    TStruct(
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

  def rowPType(requestedType: Type): PType = {
    val (leftStruct, rightStruct) = splitRequestedTypes(requestedType)
    specLeft.decodedPType(leftStruct).asInstanceOf[PStruct].insertFields(specRight.decodedPType(rightStruct).asInstanceOf[PStruct].fields.map(f => (f.name, f.typ)))
  }

  def fullRowType: TStruct = specLeft.encodedVirtualType.asInstanceOf[TStruct] ++ specRight.encodedVirtualType.asInstanceOf[TStruct]

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    partitionRegion: Value[Region],
    requestedType: Type): IEmitCode = {

    val mb = cb.emb

    val (leftRType, rightRType) = splitRequestedTypes(requestedType)

    val (leftPType: PStruct, makeLeftDec) = specLeft.buildDecoder(ctx, leftRType)
    val (rightPType: PStruct, makeRightDec) = specRight.buildDecoder(ctx, rightRType)

    // copied from TableNativeReader, but hard to pass it through to here and slightly different signature
    // plan is to remove the interpreted readers when we can evaluate TableStage to TableValue
    def fieldInserter(ctx: ExecuteContext, pLeft: PStruct, pRight: PStruct): (PStruct, Function3[FS, java.lang.Integer, Region, AsmFunction3RegionLongLongLong]) = {
      val (Some(PTypeReferenceSingleCodeType(t: PStruct)), mk) = ir.Compile[AsmFunction3RegionLongLongLong](ctx,
        FastIndexedSeq("left" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType((pLeft))), "right" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pRight))),
        FastIndexedSeq(typeInfo[Region], LongInfo, LongInfo), LongInfo,
        InsertFields(Ref("left", pLeft.virtualType),
          pRight.fieldNames.map(f =>
            f -> GetField(Ref("right", pRight.virtualType), f))))
      (t, { (fs: FS, pidx: java.lang.Integer, r) => mk(fs, pidx, r) })
    }

    val (eltType: PStruct, makeInserter) = fieldInserter(ctx, leftPType, rightPType)
    val makeInserterCode = mb.getObject[Function3[FS, java.lang.Integer, Region, AsmFunction3RegionLongLongLong]](makeInserter)

    val makeIndexCode = indexSpecLeft.map { indexSpec =>
      val (keyType, annotationType) = indexSpec.types
      val (leafPType: PStruct, leafDec) = indexSpec.leafCodec.buildDecoder(ctx, indexSpec.leafCodec.encodedVirtualType)
      val (intPType: PStruct, intDec) = indexSpec.internalNodeCodec.buildDecoder(ctx, indexSpec.internalNodeCodec.encodedVirtualType)
      val mkIndexReader = IndexReaderBuilder.withDecoders(leafDec, intDec, keyType, annotationType, leafPType, intPType)

      mb.getObject[Function4[FS, String, Int, RegionPool, IndexReader]](mkIndexReader)
    }
    val makeLeftDecCode = mb.getObject[(InputStream => Decoder)](makeLeftDec)
    val makeRightDecCode = mb.getObject[(InputStream => Decoder)](makeRightDec)

    val leftOffsetField = indexSpecLeft.flatMap(_.offsetField)
    val rightOffsetField = indexSpecRight.flatMap(_.offsetField)

    context.toI(cb).map(cb) { ctxStruct =>

      def getIndexReader(cb: EmitCodeBuilder, ctxMemo: SBaseStructValue): Code[IndexReader] = {
        makeIndexCode match {
          case Some(makeIndex) =>
            val indexPath = ctxMemo
              .loadField(cb, "indexPath")
              .handle(cb, cb._fatal(""))
              .asString
              .loadString()
            Code.checkcast[IndexReader](
              makeIndex.invoke[AnyRef, AnyRef, AnyRef, AnyRef, AnyRef]("apply", mb.getFS, indexPath, Code.boxInt(8), cb.emb.ecb.pool()))
          case None =>
            Code._null[IndexReader]
        }
      }

      def getInterval(cb: EmitCodeBuilder, region: Value[Region], ctxMemo: SBaseStructValue): Code[Interval] = {
        makeIndexCode match {
          case Some(_) =>
            ctxMemo.loadField(cb, "interval")
              .consumeCode[Interval](cb,
                Code._fatal[Interval](""),
                { pc =>
                  val pcm = pc.memoize(cb, "pnri_interval")
                  val pt = pcm.st.canonicalPType()
                  Code.invokeScalaObject2[PType, Long, Interval](
                    PartitionBoundOrdering.getClass,
                    "regionValueToJavaObject",
                    mb.getPType(pt),
                    pt.store(cb, region, pcm, false))
                }
              )
          case None => Code._null[Interval]
        }
      }

      val next = mb.genFieldThisRef[Long]("pnr_next")
      val it = mb.genFieldThisRef[MaybeIndexedReadZippedIterator]("pnri_idx_iterator")
      val region = mb.genFieldThisRef[Region]("pnri_region")

      val producer = new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          val ctxMemo = ctxStruct.asBaseStruct.memoize(cb, "pnri_ctx_struct")
          cb.assign(it,
            Code.newInstance11[MaybeIndexedReadZippedIterator,
              (InputStream) => Decoder,
              (InputStream) => Decoder,
              AsmFunction3RegionLongLongLong,
              Region,
              InputStream,
              InputStream,
              IndexReader,
              String,
              String,
              Interval,
              InputMetrics](
              makeLeftDecCode,
              makeRightDecCode,
              Code.checkcast[AsmFunction3RegionLongLongLong](makeInserterCode.invoke[AnyRef, AnyRef, AnyRef, AnyRef]("apply", cb.emb.getFS, Code.boxInt(0), region)),
              region,
              mb.open(ctxMemo.loadField(cb, "leftPartitionPath")
                .handle(cb, cb._fatal(""))
                .asString
                .loadString(), true),
              mb.open(ctxMemo.loadField(cb, "rightPartitionPath")
                .handle(cb, cb._fatal(""))
                .asString
                .loadString(), true),
              getIndexReader(cb, ctxMemo),
              leftOffsetField.map[Code[String]](const(_)).getOrElse(Code._null[String]),
              rightOffsetField.map[Code[String]](const(_)).getOrElse(Code._null[String]),
              getInterval(cb, region, ctxMemo),
              Code._null[InputMetrics]
            ))
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(next, it.invoke[Long]("_next"))
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, eltType.loadCheapPCode(cb, next)))

        override def close(cb: EmitCodeBuilder): Unit = cb += it.invoke[Unit]("close")
      }
      SStreamCode(producer)
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
) extends TableReader {
  def pathsUsed: Seq[String] = Array(params.path)

  val filterIntervals: Boolean = params.options.map(_.filterIntervals).getOrElse(false)

  def partitionCounts: Option[IndexedSeq[Long]] = if (params.options.isDefined) None else Some(spec.partitionCounts)

  def fullType: TableType = spec.table_type

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    coerce[PStruct](spec.rowsComponent.rvdSpec(ctx.fs, params.path)
      .typedCodecSpec.encodedType.decodedPType(requestedType.rowType)) ->
      coerce[PStruct](spec.globalsComponent.rvdSpec(ctx.fs, params.path)
        .typedCodecSpec.encodedType.decodedPType(requestedType.globalType))
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val (globalType, globalsOffset) = spec.globalsComponent.readLocalSingleRow(ctx, params.path, tr.typ.globalType)
    val rvd = if (tr.dropRows) {
      RVD.empty(tr.typ.canonicalRVDType)
    } else {
      val partitioner = if (filterIntervals)
        params.options.map(opts => RVDPartitioner.union(tr.typ.keyType, opts.intervals, tr.typ.key.length - 1))
      else
        params.options.map(opts => new RVDPartitioner(tr.typ.keyType, opts.intervals))
      val rvd = spec.rowsComponent.read(ctx, params.path, tr.typ.rowType, partitioner, filterIntervals)
      if (rvd.typ.key startsWith tr.typ.key)
        rvd
      else {
        log.info("Sorting a table after read. Rewrite the table to prevent this in the future.")
        rvd.changeKey(ctx, tr.typ.key)
      }
    }
    TableValue(ctx, tr.typ, BroadcastRow(ctx, RegionValue(ctx.r, globalsOffset), globalType.setRequired(true).asInstanceOf[PStruct]), rvd)
  }

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

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globalsSpec = spec.globalsSpec
    val globalsPath = spec.globalsComponent.absolutePath(params.path)
    ArrayRef(ToArray(ReadPartition(Str(globalsSpec.absolutePartPaths(globalsPath).head), requestedGlobalsType, PartitionNativeReader(globalsSpec.typedCodecSpec))), 0)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globals = lowerGlobals(ctx, requestedType.globalType)
    val rowsSpec = spec.rowsSpec
    val specPart = rowsSpec.partitioner
    val partitioner = if (filterIntervals)
      params.options.map(opts => RVDPartitioner.union(specPart.kType, opts.intervals, specPart.kType.size - 1))
    else
      params.options.map(opts => new RVDPartitioner(specPart.kType, opts.intervals))

    spec.rowsSpec.readTableStage(ctx, spec.rowsComponent.absolutePath(params.path), requestedType, partitioner, filterIntervals).apply(globals)
  }
}

case class TableNativeZippedReader(
  pathLeft: String,
  pathRight: String,
  options: Option[NativeReaderOptions],
  specLeft: AbstractTableSpec,
  specRight: AbstractTableSpec
) extends TableReader {
  def pathsUsed: Seq[String] = FastSeq(pathLeft, pathRight)

  override def renderShort(): String = s"(TableNativeZippedReader $pathLeft $pathRight ${ options.map(_.renderShort()).getOrElse("") })"

  private lazy val filterIntervals = options.exists(_.filterIntervals)

  private def intervals = options.map(_.intervals)

  require((specLeft.table_type.rowType.fieldNames ++ specRight.table_type.rowType.fieldNames).areDistinct())
  require(specRight.table_type.key.isEmpty)
  require(specLeft.partitionCounts sameElements specRight.partitionCounts)
  require(specLeft.version == specRight.version)

  def partitionCounts: Option[IndexedSeq[Long]] = if (intervals.isEmpty) Some(specLeft.partitionCounts) else None

  override lazy val fullType: TableType = specLeft.table_type.copy(rowType = specLeft.table_type.rowType ++ specRight.table_type.rowType)
  private val leftFieldSet = specLeft.table_type.rowType.fieldNames.toSet
  private val rightFieldSet = specRight.table_type.rowType.fieldNames.toSet

  def leftRType(requestedType: TStruct): TStruct =
    requestedType.filter(f => leftFieldSet.contains(f.name))._1

  def rightRType(requestedType: TStruct): TStruct =
    requestedType.filter(f => rightFieldSet.contains(f.name))._1

  def leftPType(ctx: ExecuteContext, leftRType: TStruct): PStruct =
    coerce[PStruct](specLeft.rowsComponent.rvdSpec(ctx.fs, pathLeft)
      .typedCodecSpec.encodedType.decodedPType(leftRType))

  def rightPType(ctx: ExecuteContext, rightRType: TStruct): PStruct =
    coerce[PStruct](specRight.rowsComponent.rvdSpec(ctx.fs, pathRight)
      .typedCodecSpec.encodedType.decodedPType(rightRType))

  def rowAndGlobalPTypes(ctx: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    fieldInserter(ctx, leftPType(ctx, leftRType(requestedType.rowType)),
      rightPType(ctx, rightRType(requestedType.rowType)))._1 ->
      coerce[PStruct](specLeft.globalsComponent.rvdSpec(ctx.fs, pathLeft)
        .typedCodecSpec.encodedType.decodedPType(requestedType.globalType))
  }

  def fieldInserter(ctx: ExecuteContext, pLeft: PStruct, pRight: PStruct): (PStruct, (FS, Int, Region) => AsmFunction3RegionLongLongLong) = {
    val (Some(PTypeReferenceSingleCodeType(t: PStruct)), mk) = ir.Compile[AsmFunction3RegionLongLongLong](ctx,
      FastIndexedSeq("left" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pLeft)), "right" -> SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pRight))),
      FastIndexedSeq(typeInfo[Region], LongInfo, LongInfo), LongInfo,
      InsertFields(Ref("left", pLeft.virtualType),
        pRight.fieldNames.map(f =>
          f -> GetField(Ref("right", pRight.virtualType), f))))
    (t, mk)
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val fs = ctx.fs
    val (globalPType: PStruct, globalsOffset) = specLeft.globalsComponent.readLocalSingleRow(ctx, pathLeft, tr.typ.globalType)
    val rvd = if (tr.dropRows) {
      RVD.empty(tr.typ.canonicalRVDType)
    } else {
      val partitioner = if (filterIntervals)
        intervals.map(i => RVDPartitioner.union(tr.typ.keyType, i, tr.typ.key.length - 1))
      else
        intervals.map(i => new RVDPartitioner(tr.typ.keyType, i))
      if (tr.typ.rowType.fieldNames.forall(f => !rightFieldSet.contains(f))) {
        specLeft.rowsComponent.read(ctx, pathLeft, tr.typ.rowType, partitioner, filterIntervals)
      } else if (tr.typ.rowType.fieldNames.forall(f => !leftFieldSet.contains(f))) {
        specRight.rowsComponent.read(ctx, pathRight, tr.typ.rowType, partitioner, filterIntervals)
      } else {
        val rvdSpecLeft = specLeft.rowsComponent.rvdSpec(fs, pathLeft)
        val rvdSpecRight = specRight.rowsComponent.rvdSpec(fs, pathRight)
        val rvdPathLeft = specLeft.rowsComponent.absolutePath(pathLeft)
        val rvdPathRight = specRight.rowsComponent.absolutePath(pathRight)

        val leftRType = tr.typ.rowType.filter(f => leftFieldSet.contains(f.name))._1
        val rightRType = tr.typ.rowType.filter(f => rightFieldSet.contains(f.name))._1

        AbstractRVDSpec.readZipped(ctx,
          rvdSpecLeft, rvdSpecRight,
          rvdPathLeft, rvdPathRight,
          partitioner, filterIntervals,
          tr.typ.rowType,
          leftRType, rightRType,
          tr.typ.key,
          fieldInserter)
      }
    }

    TableValue(ctx, tr.typ, BroadcastRow(ctx, RegionValue(ctx.r, globalsOffset), globalPType.setRequired(true).asInstanceOf[PStruct]), rvd)
  }

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR = {
    val globalsSpec = specLeft.globalsSpec
    val globalsPath = specLeft.globalsComponent.absolutePath(pathLeft)
    ArrayRef(ToArray(ReadPartition(Str(globalsSpec.absolutePartPaths(globalsPath).head), requestedGlobalsType, PartitionNativeReader(globalsSpec.typedCodecSpec))), 0)
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globals = lowerGlobals(ctx, requestedType.globalType)
    val rowsSpec = specLeft.rowsSpec
    val specPart = rowsSpec.partitioner
    val partitioner = if (filterIntervals)
      options.map(opts => RVDPartitioner.union(specPart.kType, opts.intervals, specPart.kType.size - 1))
    else
      options.map(opts => new RVDPartitioner(specPart.kType, opts.intervals))

    def splitRequestedTypes(requestedType: Type): (TStruct, TStruct) = {
      val reqStruct = requestedType.asInstanceOf[TStruct]
      val neededFields = reqStruct.fieldNames.toSet

      val leftProvidedFields = specLeft.table_type.rowType.fieldNames.toSet
      val rightProvidedFields = specRight.table_type.rowType.fieldNames.toSet
      val leftNeededFields = leftProvidedFields.intersect(neededFields)
      val rightNeededFields = rightProvidedFields.intersect(neededFields)
      assert(leftNeededFields.intersect(rightNeededFields).isEmpty)

      val leftStruct = reqStruct.filterSet(leftNeededFields)._1
      val rightStruct = reqStruct.filterSet(rightNeededFields)._1
      (leftStruct, rightStruct)
    }

    val (reqLeft, reqRight) = splitRequestedTypes(requestedType.rowType)

    AbstractRVDSpec.readZippedLowered(ctx,
      specLeft.rowsSpec, specRight.rowsSpec,
      pathLeft + "/rows", pathRight + "/rows",
      partitioner, filterIntervals,
      requestedType.rowType, reqLeft, reqRight, requestedType.key).apply(globals)
  }

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

case class TableFromBlockMatrixNativeReader(params: TableFromBlockMatrixNativeReaderParameters, metadata: BlockMatrixMetadata) extends TableReader {
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

  override lazy val fullType: TableType = {
    val rowType = TStruct("row_idx" -> TInt64, "entries" -> TArray(TFloat64))
    TableType(rowType, Array("row_idx"), TStruct.empty)
  }

  def rowAndGlobalPTypes(context: ExecuteContext, tableType: TableType): (PStruct, PStruct) = {
    PType.canonical(tableType.rowType, required = true).asInstanceOf[PStruct] ->
      PCanonicalStruct.empty(required = true)
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val rowsRDD = new BlockMatrixReadRowBlockedRDD(ctx.fsBc, params.path, partitionRanges, metadata,
      maybeMaximumCacheMemoryInBytes = params.maximumCacheMemoryInBytes)

    val partitionBounds = partitionRanges.map { r => Interval(Row(r.start), Row(r.end), true, false) }
    val partitioner = new RVDPartitioner(fullType.keyType, partitionBounds)

    val rowTyp = rowAndGlobalPTypes(ctx, tr.typ)._1
    val rvd = RVD(RVDType(rowTyp, fullType.key.filter(rowTyp.hasField)), partitioner, ContextRDD(rowsRDD))
    TableValue(ctx, fullType, BroadcastRow.empty(ctx), rvd)
  }

  override def toJValue: JValue = {
    decomposeWithName(params, "TableFromBlockMatrixNativeReader")(TableReader.formats)
  }

  def renderShort(): String = defaultRender()
}

object TableRead {
  def native(fs: FS, path: String): TableRead = {
    val tr = TableNativeReader(fs, TableNativeReaderParameters(path, None))
    TableRead(tr.fullType, false, tr)
  }
}

case class TableRead(typ: TableType, dropRows: Boolean, tr: TableReader) extends TableIR {
  assert(PruneDeadFields.isSupertype(typ, tr.fullType),
    s"\n  original:  ${ tr.fullType }\n  requested: $typ")

  override def partitionCounts: Option[IndexedSeq[Long]] = if (dropRows) Some(FastIndexedSeq(0L)) else tr.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = partitionCounts.map(_.sum)

  val children: IndexedSeq[BaseIR] = Array.empty[BaseIR]

  def copy(newChildren: IndexedSeq[BaseIR]): TableRead = {
    assert(newChildren.isEmpty)
    TableRead(typ, dropRows, tr)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = tr.apply(this, ctx)
}

case class TableParallelize(rowsAndGlobal: IR, nPartitions: Option[Int] = None) extends TableIR {
  require(rowsAndGlobal.typ.isInstanceOf[TStruct])
  require(rowsAndGlobal.typ.asInstanceOf[TStruct].fieldNames.sameElements(Array("rows", "global")))

  lazy val rowCountUpperBound: Option[Long] = None

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

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val (ptype: PStruct, res) = CompileAndEvaluate._apply(ctx, rowsAndGlobal, optimize = false) match {
      case Right((t, off)) => (t.fields(0).typ, t.loadField(off, 0))
    }

    val globalsT = ptype.types(1).setRequired(true).asInstanceOf[PStruct]
    if (ptype.isFieldMissing(res, 1))
      fatal("'parallelize': found missing global value")
    val globals = BroadcastRow(ctx, RegionValue(ctx.r, ptype.loadField(res, 1)), globalsT)

    val rowsT = ptype.types(0).asInstanceOf[PArray]
    val rowT = rowsT.elementType.asInstanceOf[PStruct].setRequired(true)
    val spec = TypedCodecSpec(rowT, BufferSpec.wireSpec)

    val makeEnc = spec.buildEncoder(ctx, rowT)
    val rowsAddr = ptype.loadField(res, 0)
    val nRows = rowsT.loadLength(rowsAddr)

    val nSplits = math.min(nPartitions.getOrElse(16), math.max(nRows, 1))
    val parts = partition(nRows, nSplits)

    val bae = new ByteArrayEncoder(makeEnc)
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

    val (resultRowType: PStruct, makeDec) = spec.buildDecoder(ctx, typ.rowType)
    assert(resultRowType.virtualType == typ.rowType, s"typ mismatch:" +
      s"\n  res=${ resultRowType.virtualType }\n  typ=${ typ.rowType }")

    log.info(s"parallelized $nRows rows in $nSplits partitions")

    val rvd = ContextRDD.parallelize(encRows, encRows.length)
      .cmapPartitions { (ctx, it) =>
        it.flatMap { case (nRowPartition, arr) =>
          val bais = new ByteArrayDecoder(makeDec)
          bais.set(arr)
          Iterator.range(0, nRowPartition)
            .map { _ =>
              bais.readValue(ctx.region)
            }
        }
      }
    TableValue(ctx, typ, globals, RVD.unkeyed(resultRowType, rvd))
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val children: IndexedSeq[BaseIR] = Array(child)

  val typ: TableType = child.typ.copy(key = keys)

  def definitelyDoesNotShuffle: Boolean = child.typ.key.startsWith(keys) || isSorted

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyBy = {
    assert(newChildren.length == 1)
    TableKeyBy(newChildren(0).asInstanceOf[TableIR], keys, isSorted)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)
    tv.copy(typ = typ, rvd = tv.rvd.enforceKey(ctx, keys, isSorted))
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

  lazy val rowCountUpperBound: Option[Long] = Some(n.toLong)

  val typ: TableType = TableType(
    TStruct("idx" -> TInt32),
    Array("idx"),
    TStruct.empty)

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val localRowType = PCanonicalStruct(true, "idx" -> PInt32Required)
    val localPartCounts = partCounts
    val partStarts = partCounts.scanLeft(0)(_ + _)
    TableValue(ctx, typ,
      BroadcastRow.empty(ctx),
      new RVD(
        RVDType(localRowType, Array("idx")),
        new RVDPartitioner(Array("idx"), typ.rowType,
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
          }))
  }
}

case class TableFilter(child: TableIR, pred: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, pred)

  val typ: TableType = child.typ

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableFilter = {
    assert(newChildren.length == 2)
    TableFilter(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)

    if (pred == True())
      return tv
    else if (pred == False())
      return tv.copy(rvd = RVD.empty(typ.canonicalRVDType))

    val (Some(BooleanSingleCodeType), f) = ir.Compile[AsmFunction3RegionLongLongBoolean](
      ctx,
      FastIndexedSeq(("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType))),
        ("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), BooleanInfo,
      Coalesce(FastIndexedSeq(pred, False())))

    tv.filterWithPartitionOp(ctx.fsBc, f)((rowF, ctx, ptr, globalPtr) => rowF(ctx.region, ptr, globalPtr))
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

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    child.partitionCounts.map(subsetKind match {
      case TableSubset.HEAD => PartitionCounts.getHeadPCs(_, n)
      case TableSubset.TAIL => PartitionCounts.getTailPCs(_, n)
    })

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound match {
    case Some(c) => Some(c.min(n))
    case None => Some(n)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)
    prev.copy(rvd = subsetKind match {
      case TableSubset.HEAD => prev.rvd.head(n, child.partitionCounts)
      case TableSubset.TAIL => prev.rvd.tail(n, child.partitionCounts)
    })
  }
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

  lazy val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableRepartition = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableRepartition(newChild, n, strategy)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)
    val rvd = strategy match {
      case RepartitionStrategy.SHUFFLE => prev.rvd.coalesce(ctx, n, shuffle = true)
      case RepartitionStrategy.COALESCE => prev.rvd.coalesce(ctx, n, shuffle = false)
      case RepartitionStrategy.NAIVE_COALESCE => prev.rvd.naiveCoalesce(n, ctx)
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
    joinType == "outer")

  val children: IndexedSeq[BaseIR] = Array(left, right)

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

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val leftTV = left.execute(ctx)
    val rightTV = right.execute(ctx)

    val combinedRow = Row.fromSeq(leftTV.globals.javaValue.toSeq ++ rightTV.globals.javaValue.toSeq)
    val newGlobals = BroadcastRow(ctx, combinedRow, newGlobalType)

    val leftRVDType = leftTV.rvd.typ.copy(key = left.typ.key.take(joinKey))
    val rightRVDType = rightTV.rvd.typ.copy(key = right.typ.key.take(joinKey))

    val leftRowType = leftRVDType.rowType
    val rightRowType = rightRVDType.rowType
    val leftKeyFieldIdx = leftRVDType.kFieldIdx
    val rightKeyFieldIdx = rightRVDType.kFieldIdx
    val leftValueFieldIdx = leftRVDType.valueFieldIdx
    val rightValueFieldIdx = rightRVDType.valueFieldIdx

    def noIndex(pfs: IndexedSeq[PField]): IndexedSeq[(String, PType)] =
      pfs.map(pf => (pf.name, pf.typ))

    def unionFieldPTypes(ps: PStruct, ps2: PStruct): IndexedSeq[(String, PType)] =
      ps.fields.zip(ps2.fields).map { case (pf1, pf2) =>
        (pf1.name, InferPType.getCompatiblePType(Seq(pf1.typ, pf2.typ)))
      }

    def castFieldRequiredeness(ps: PStruct, required: Boolean): IndexedSeq[(String, PType)] =
      ps.fields.map(pf => (pf.name, pf.typ.setRequired(required)))

    val (lkT, lvT, rvT) = joinType match {
      case "inner" =>
        val keyTypeFields = castFieldRequiredeness(leftRVDType.kType, true)
        (keyTypeFields, noIndex(leftRVDType.valueType.fields), noIndex(rightRVDType.valueType.fields))
      case "left" =>
        val rValueTypeFields = castFieldRequiredeness(rightRVDType.valueType, false)
        (noIndex(leftRVDType.kType.fields), noIndex(leftRVDType.valueType.fields), rValueTypeFields)
      case "right" =>
        val keyTypeFields = leftRVDType.kType.fields.zip(rightRVDType.kType.fields).map({
          case (pf1, pf2) => {
            assert(pf1.typ isOfType pf2.typ)
            (pf1.name, pf2.typ)
          }
        })
        val lValueTypeFields = castFieldRequiredeness(leftRVDType.valueType, false)
        (keyTypeFields, lValueTypeFields, noIndex(rightRVDType.valueType.fields))
      case "outer" =>
        val keyTypeFields = unionFieldPTypes(leftRVDType.kType, rightRVDType.kType)
        val lValueTypeFields = castFieldRequiredeness(leftRVDType.valueType, false)
        val rValueTypeFields = castFieldRequiredeness(rightRVDType.valueType, false)
        (keyTypeFields, lValueTypeFields, rValueTypeFields)
    }

    val newRowPType = PCanonicalStruct(true, lkT ++ lvT ++ rvT: _*)

    assert(newRowPType.virtualType == newRowType)

    val rvMerger = { (_: RVDContext, it: Iterator[JoinedRegionValue]) =>
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

        rvb.start(newRowPType)
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

    val leftRVD = leftTV.rvd
    val rightRVD = rightTV.rvd
    val joinedRVD = leftRVD.orderedJoin(
      rightRVD,
      joinKey,
      joinType,
      rvMerger,
      RVDType(newRowPType, newKey),
      ctx)

    TableValue(ctx, typ, newGlobals, joinedRVD)
  }
}

case class TableIntervalJoin(
  left: TableIR,
  right: TableIR,
  root: String,
  product: Boolean
) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  lazy val rowCountUpperBound: Option[Long] = left.rowCountUpperBound

  val rightType: Type = if (product) TArray(right.typ.valueType) else right.typ.valueType
  val typ: TableType = left.typ.copy(rowType = left.typ.rowType.appendKey(root, rightType))

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR =
    TableIntervalJoin(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[TableIR], root, product)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val leftValue = left.execute(ctx)
    val rightValue = right.execute(ctx)

    val leftRVDType = leftValue.rvd.typ
    val rightRVDType = rightValue.rvd.typ.copy(key = rightValue.typ.key)
    val rightValueFields = rightRVDType.valueType.fieldNames

    val localKey = typ.key
    val localRoot = root
    val newRVD =
      if (product) {
        val joiner = (rightPType: PStruct) => {
          val leftRowType = leftRVDType.rowType
          val newRowType = leftRowType.appendKey(localRoot, PCanonicalArray(rightPType.selectFields(rightValueFields)))
          (RVDType(newRowType, localKey), (_: RVDContext, it: Iterator[Muple[RegionValue, Iterable[RegionValue]]]) => {
            val rvb = new RegionValueBuilder()
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
            val rvb = new RegionValueBuilder()
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

    TableValue(ctx, typ, leftValue.globals, newRVD)
  }
}

/**
  * The TableMultiWayZipJoin node assumes that input tables have distinct keys. If inputs
  * do not have distinct keys, the key that is included in the result is undefined, but
  * is likely the last.
  */
case class TableMultiWayZipJoin(children: IndexedSeq[TableIR], fieldName: String, globalName: String) extends TableIR {
  require(children.length > 0, "there must be at least one table as an argument")

  private val first = children.head
  private val rest = children.tail

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

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val childValues = children.map(_.execute(ctx))

    val childRVDs = RVD.unify(childValues.map(_.rvd)).toFastIndexedSeq
    assert(childRVDs.forall(_.typ.key.startsWith(typ.key)))

    val repartitionedRVDs =
      if (childRVDs(0).partitioner.satisfiesAllowedOverlap(typ.key.length - 1) &&
        childRVDs.forall(rvd => rvd.partitioner == childRVDs(0).partitioner))
        childRVDs.map(_.truncateKey(typ.key.length))
      else {
        info("TableMultiWayZipJoin: repartitioning children")
        val childRanges = childRVDs.flatMap(_.partitioner.coarsenedRangeBounds(typ.key.length))
        val newPartitioner = RVDPartitioner.generate(typ.keyType, childRanges)
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
    val localNewRowType = PCanonicalStruct(required = true,
      keyFields ++ Array((fieldName, PCanonicalArray(
        PCanonicalStruct(required = false, valueFields: _*), required = true))): _*)
    val localDataLength = children.length
    val rvMerger = { (ctx: RVDContext, it: Iterator[BoxedArrayBuilder[(RegionValue, Int)]]) =>
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
      crdd = ContextRDD.czipNPartitions(repartitionedRVDs.map(_.crdd.toCRDDRegionValue)) { (ctx, its) =>
        val orvIters = its.map(it => OrderedRVIterator(localRVDType, it, ctx))
        rvMerger(ctx, OrderedRVIterator.multiZipJoin(orvIters))
      }.toCRDDPtr)

    val newGlobals = BroadcastRow(ctx,
      Row(childValues.map(_.globals.javaValue)),
      newGlobalType)

    TableValue(ctx, typ, newGlobals, rvd)
  }
}

case class TableLeftJoinRightDistinct(left: TableIR, right: TableIR, root: String) extends TableIR {
  require(right.typ.keyType isPrefixOf left.typ.keyType,
    s"\n  L: ${ left.typ }\n  R: ${ right.typ }")

  lazy val rowCountUpperBound: Option[Long] = left.rowCountUpperBound

  lazy val children: IndexedSeq[BaseIR] = Array(left, right)

  private val newRowType = left.typ.rowType.structInsert(right.typ.valueType, List(root))._1
  val typ: TableType = left.typ.copy(rowType = newRowType)

  override def partitionCounts: Option[IndexedSeq[Long]] = left.partitionCounts

  def copy(newChildren: IndexedSeq[BaseIR]): TableLeftJoinRightDistinct = {
    val IndexedSeq(newLeft: TableIR, newRight: TableIR) = newChildren
    TableLeftJoinRightDistinct(newLeft, newRight, root)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val leftValue = left.execute(ctx)
    val rightValue = right.execute(ctx)

    val joinKey = math.min(left.typ.key.length, right.typ.key.length)
    leftValue.copy(
      typ = typ,
      rvd = leftValue.rvd
        .orderedLeftJoinDistinctAndInsert(rightValue.rvd.truncateKey(joinKey), root))
  }
}

case class TableMapPartitions(child: TableIR,
  globalName: String,
  partitionStreamName: String,
  body: IR
) extends TableIR {
  assert(body.typ.isInstanceOf[TStream], s"${ body.typ }")
  lazy val typ = child.typ.copy(
    rowType = body.typ.asInstanceOf[TStream].elementType.asInstanceOf[TStruct])

  lazy val children: IndexedSeq[BaseIR] = Array(child, body)

  val rowCountUpperBound: Option[Long] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): TableMapPartitions = {
    assert(newChildren.length == 2)
    TableMapPartitions(newChildren(0).asInstanceOf[TableIR],
      globalName, partitionStreamName, newChildren(1).asInstanceOf[IR])
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)
    val rowPType = tv.rvd.rowPType
    val globalPType = tv.globals.t

    val partitionPType = PCanonicalStream(rowPType, required = true)
    val (newRowPType: PStruct, makeIterator) = CompileIterator.forTableMapPartitions(
      ctx,
      globalPType, partitionPType,
      Subst(body, BindingEnv(Env(
        globalName -> In(0, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(globalPType))),
        partitionStreamName -> In(1, SingleCodeEmitParamType(true, StreamSingleCodeType(requiresMemoryManagementPerElement = true, rowPType)))))))

    val globalsOff = tv.globals.value.offset

    val fsBc = tv.ctx.fsBc
    val itF = { (idx: Int, consumerCtx: RVDContext, partition: (RVDContext) => Iterator[Long]) =>
      val boxedPartition = new StreamArgType {
        def apply(outerRegion: Region, eltRegion: Region): Iterator[java.lang.Long] =
          partition(new RVDContext(outerRegion, eltRegion)).map(box)
      }
      makeIterator(fsBc.value, idx, consumerCtx,
        globalsOff,
        boxedPartition
      ).map(l => l.longValue())
    }

    tv.copy(
      typ = typ,
      rvd = tv.rvd
        .mapPartitionsWithContextAndIndex(RVDType(newRowPType, typ.key))(itF))
  }
}

// Must leave key fields unchanged.
case class TableMapRows(child: TableIR, newRow: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newRow)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val typ: TableType = child.typ.copy(rowType = newRow.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapRows = {
    assert(newChildren.length == 2)
    TableMapRows(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)
    val fsBc = ctx.fsBc

    val scanRef = genUID()
    val extracted = agg.Extract.apply(newRow, scanRef, Requiredness(this, ctx), isScan = true)

    if (extracted.aggs.isEmpty) {
      val (Some(PTypeReferenceSingleCodeType(rTyp)), f) = ir.Compile[AsmFunction3RegionLongLongLong](
        ctx,
        FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
          ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))),
        FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
        Coalesce(FastIndexedSeq(
          extracted.postAggIR,
          Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ))))

      val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, "global")
      val globalsBc =
        if (rowIterationNeedsGlobals)
          tv.globals.broadcast
        else
          null

      val fsBc = ctx.fsBc
      val itF = { (i: Int, ctx: RVDContext, it: Iterator[Long]) =>
        val globalRegion = ctx.partitionRegion
        val globals = if (rowIterationNeedsGlobals)
          globalsBc.value.readRegionValue(globalRegion)
        else
          0

        val newRow = f(fsBc.value, i, globalRegion)
        it.map { ptr =>
          newRow(ctx.r, globals, ptr)
        }
      }

      return tv.copy(
        typ = typ,
        rvd = tv.rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key))(itF))
    }

    val scanInitNeedsGlobals = Mentions(extracted.init, "global")
    val scanSeqNeedsGlobals = Mentions(extracted.seqPerElt, "global")
    val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, "global")

    val globalsBc =
      if (rowIterationNeedsGlobals || scanInitNeedsGlobals || scanSeqNeedsGlobals)
        tv.globals.broadcast
      else
        null

    val spec = BufferSpec.defaultUncompressed

    // Order of operations:
    // 1. init op on all aggs and serialize to byte array.
    // 2. load in init op on each partition, seq op over partition, serialize.
    // 3. load in partition aggregations, comb op as necessary, serialize.
    // 4. load in partStarts, calculate newRow based on those results.

    val (_, initF) = ir.CompileWithAggregators[AsmFunction2RegionLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
      Begin(FastIndexedSeq(extracted.init)))

    val serializeF = extracted.serialize(ctx, spec)

    val (_, eltSeqF) = ir.CompileWithAggregators[AsmFunction3RegionLongLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
        ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
      extracted.eltOp(ctx))

    val read = extracted.deserialize(ctx, spec)
    val write = extracted.serialize(ctx, spec)
    val combOpFNeedsPool = extracted.combOpFSerializedFromRegionPool(ctx, spec)

    val (Some(PTypeReferenceSingleCodeType(rTyp)), f) = ir.CompileWithAggregators[AsmFunction3RegionLongLongLong](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t))),
        ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.rvd.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
      Let(scanRef, extracted.results,
        Coalesce(FastIndexedSeq(
          extracted.postAggIR,
          Die("Internal error: TableMapRows: row expression missing", extracted.postAggIR.typ)))))
    assert(rTyp.virtualType == newRow.typ)

    // 1. init op on all aggs and write out to initPath
    val initAgg = ctx.r.pool.scopedRegion { aggRegion =>
      ctx.r.pool.scopedRegion { fRegion =>
        val init = initF(fsBc.value, 0, fRegion)
        init.newAggState(aggRegion)
        init(fRegion, tv.globals.value.offset)
        serializeF(aggRegion, init.getAggOffset())
      }
    }

    if (HailContext.getFlag("distributed_scan_comb_op") != null) {
      val fsBc = ctx.fs.broadcast
      val tmpBase = ctx.createTmpPath("table-map-rows-distributed-scan")
      val d = digitsNeeded(tv.rvd.getNumPartitions)
      val files = tv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
        val path = tmpBase + "/" + partFile(d, i, TaskContext.get)
        val globalRegion = ctx.freshRegion()
        val globals = if (scanSeqNeedsGlobals) globalsBc.value.readRegionValue(globalRegion) else 0

        ctx.r.pool.scopedSmallRegion { aggRegion =>
          val seq = eltSeqF(fsBc.value, i, globalRegion)

          seq.setAggState(aggRegion, read(aggRegion, initAgg))
          it.foreach { ptr =>
            seq(ctx.region, globals, ptr)
            ctx.region.clear()
          }
          using(new DataOutputStream(fsBc.value.create(path))) { os =>
            val bytes = write(aggRegion, seq.getAggOffset())
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
        log.info(s"Running combOp stage with $nToMerge tasks")
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
              val bytes = combOpFNeedsPool(() => ctx.r.pool)(b1, b2)
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
          globalsBc.value.readRegionValue(globalRegion)
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

            b = combOpFNeedsPool(() => ctx.r.pool)(b, using(new DataInputStream(fsBc.value.open(path)))(readToBytes))
          }
          b
        }

        val aggRegion = ctx.freshRegion()
        val newRow = f(fsBc.value, i, globalRegion)
        val seq = eltSeqF(fsBc.value, i, globalRegion)
        var aggOff = read(aggRegion, partitionAggs)

        val res = it.map { ptr =>
          newRow.setAggState(aggRegion, aggOff)
          val newPtr = newRow(ctx.region, globals, ptr)
          seq.setAggState(aggRegion, newRow.getAggOffset())
          seq(ctx.region, globals, ptr)
          aggOff = seq.getAggOffset()
          newPtr
        }
        aggRegion.invalidate()

        res
      }
      return tv.copy(
        typ = typ,
        rvd = tv.rvd.mapPartitionsWithIndex(RVDType(rTyp.asInstanceOf[PStruct], typ.key))(itF))
    }

    // 2. load in init op on each partition, seq op over partition, write out.
    val scanPartitionAggs = SpillingCollectIterator(ctx.localTmpdir, ctx.fs, tv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
      val globalRegion = ctx.partitionRegion
      val globals = if (scanSeqNeedsGlobals) globalsBc.value.readRegionValue(globalRegion) else 0

      SparkTaskContext.get().getRegionPool().scopedSmallRegion { aggRegion =>
        val seq = eltSeqF(fsBc.value, i, globalRegion)

        seq.setAggState(aggRegion, read(aggRegion, initAgg))
        it.foreach { ptr =>
          seq(ctx.region, globals, ptr)
          ctx.region.clear()
        }
        Iterator.single(write(aggRegion, seq.getAggOffset()))
      }
    }, HailContext.getFlag("max_leader_scans").toInt)

    // 3. load in partition aggregations, comb op as necessary, write back out.
    val partAggs = scanPartitionAggs.scanLeft(initAgg)(combOpFNeedsPool(() => ctx.r.pool))
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
      val globals = if (rowIterationNeedsGlobals || scanSeqNeedsGlobals)
        globalsBc.value.readRegionValue(globalRegion)
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
      val newRow = f(fsBc.value, i, globalRegion)
      val seq = eltSeqF(fsBc.value, i, globalRegion)
      var aggOff = read(aggRegion, partitionAggs)

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
    tv.copy(
      typ = typ,
      rvd = tv.rvd.mapPartitionsWithIndexAndValue(RVDType(rTyp.asInstanceOf[PStruct], typ.key), partitionIndices)(itF))
  }
}

case class TableMapGlobals(child: TableIR, newGlobals: IR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child, newGlobals)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val typ: TableType =
    child.typ.copy(globalType = newGlobals.typ.asInstanceOf[TStruct])

  def copy(newChildren: IndexedSeq[BaseIR]): TableMapGlobals = {
    assert(newChildren.length == 2)
    TableMapGlobals(newChildren(0).asInstanceOf[TableIR], newChildren(1).asInstanceOf[IR])
  }

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)

    val (Some(PTypeReferenceSingleCodeType(resultPType: PStruct)), f) = Compile[AsmFunction2RegionLongLong](ctx,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(tv.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
      Coalesce(FastIndexedSeq(
        newGlobals,
        Die("Internal error: TableMapGlobals: globals missing", newGlobals.typ))))

    val resultOff = f(ctx.fs, 0, ctx.r)(ctx.r, tv.globals.value.offset)
    tv.copy(typ = typ,
      globals = BroadcastRow(ctx, RegionValue(ctx.r, resultOff), resultPType))
  }
}

case class TableExplode(child: TableIR, path: IndexedSeq[String]) extends TableIR {
  assert(path.nonEmpty)
  assert(!child.typ.key.contains(path.head))

  lazy val rowCountUpperBound: Option[Long] = None

  lazy val children: IndexedSeq[BaseIR] = Array(child)

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
      Ref(genUID(), coerce[TStruct](struct.typ).field(name).typ))

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

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)

    val (len, l) = Compile[AsmFunction2RegionLongInt](ctx,
      FastIndexedSeq(("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo), IntInfo,
      length)
    val (Some(PTypeReferenceSingleCodeType(newRowType: PStruct)), f) = Compile[AsmFunction3RegionLongIntLong](
      ctx,
      FastIndexedSeq(("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType))),
        (idx.name, SingleCodeEmitParamType(true, Int32SingleCodeType))),
      FastIndexedSeq(classInfo[Region], LongInfo, IntInfo), LongInfo,
      newRow)
    assert(newRowType.virtualType == typ.rowType)

    val rvdType: RVDType = RVDType(
      newRowType,
      prev.rvd.typ.key.takeWhile(_ != path.head)
    )
    val fsBc = ctx.fsBc
    TableValue(ctx, typ,
      prev.globals,
      prev.rvd.boundary.mapPartitionsWithIndex(rvdType) { (i, ctx, it) =>
        val globalRegion = ctx.partitionRegion
        val lenF = l(fsBc.value, i, globalRegion)
        val rowF = f(fsBc.value, i, globalRegion)
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
  }
}

case class TableUnion(children: IndexedSeq[TableIR]) extends TableIR {
  assert(children.nonEmpty)
  assert(children.tail.forall(_.typ.rowType == children(0).typ.rowType))
  assert(children.tail.forall(_.typ.key == children(0).typ.key))

  lazy val rowCountUpperBound: Option[Long] = {
    val definedChildren = children.flatMap(_.rowCountUpperBound)
    if (definedChildren.length == children.length)
      Some(definedChildren.sum)
    else
      None
  }

  def copy(newChildren: IndexedSeq[BaseIR]): TableUnion = {
    TableUnion(newChildren.map(_.asInstanceOf[TableIR]))
  }

  val typ: TableType = children(0).typ

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tvs = children.map(_.execute(ctx))
    tvs(0).copy(
      rvd = RVD.union(RVD.unify(tvs.map(_.rvd)), tvs(0).typ.key.length, ctx))
  }
}

case class MatrixRowsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  override def partitionCounts: Option[IndexedSeq[Long]] = child.partitionCounts

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixRowsTable = {
    assert(newChildren.length == 1)
    MatrixRowsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.rowsTableType
}

case class MatrixColsTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixColsTable = {
    assert(newChildren.length == 1)
    MatrixColsTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.colsTableType
}

case class MatrixEntriesTable(child: MatrixIR) extends TableIR {
  val children: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): MatrixEntriesTable = {
    assert(newChildren.length == 1)
    MatrixEntriesTable(newChildren(0).asInstanceOf[MatrixIR])
  }

  val typ: TableType = child.typ.entriesTableType
}

case class TableDistinct(child: TableIR) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableDistinct = {
    val IndexedSeq(newChild) = newChildren
    TableDistinct(newChild.asInstanceOf[TableIR])
  }

  val typ: TableType = child.typ

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)
    prev.copy(rvd = prev.rvd.truncateKey(prev.typ.key).distinctByKey(ctx))
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableKeyByAndAggregate = {
    val IndexedSeq(newChild: TableIR, newExpr: IR, newNewKey: IR) = newChildren
    TableKeyByAndAggregate(newChild, newExpr, newNewKey, nPartitions, bufferSize)
  }

  private val keyType = newKey.typ.asInstanceOf[TStruct]
  val typ: TableType = TableType(rowType = keyType ++ coerce[TStruct](expr.typ),
    globalType = child.typ.globalType,
    key = keyType.fieldNames
  )

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)
    val fsBc = ctx.fsBc

    val localKeyType = keyType
    val (Some(PTypeReferenceSingleCodeType(localKeyPType: PStruct)), makeKeyF) = ir.Compile[AsmFunction3RegionLongLongLong](ctx,
      FastIndexedSeq(("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType))),
        ("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
      Coalesce(FastIndexedSeq(
        newKey,
        Die("Internal error: TableKeyByAndAggregate: newKey missing", newKey.typ))))

    val globalsBc = prev.globals.broadcast

    val spec = BufferSpec.defaultUncompressed
    val res = genUID()

    val extracted = agg.Extract(expr, res, Requiredness(this, ctx))

    val (_, makeInit) = ir.CompileWithAggregators[AsmFunction2RegionLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
      extracted.init)

    val (_, makeSeq) = ir.CompileWithAggregators[AsmFunction3RegionLongLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
        ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.rvd.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
      extracted.seqPerElt)

    val (Some(PTypeReferenceSingleCodeType(rTyp: PStruct)), makeAnnotate) = ir.CompileWithAggregators[AsmFunction2RegionLongLong](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
      Let(res, extracted.results, extracted.postAggIR))
    assert(rTyp.virtualType == typ.valueType, s"$rTyp, ${ typ.valueType }")

    val serialize = extracted.serialize(ctx, spec)
    val deserialize = extracted.deserialize(ctx, spec)
    val combOp = extracted.combOpFSerializedWorkersOnly(ctx, spec)

    val initF = makeInit(fsBc.value, 0, ctx.r)
    val globalsOffset = prev.globals.value.offset
    val initAggs = ctx.r.pool.scopedRegion { aggRegion =>
      initF.newAggState(aggRegion)
      initF(ctx.r, globalsOffset)
      serialize(aggRegion, initF.getAggOffset())
    }

    val newRowType = PCanonicalStruct(required = true,
      localKeyPType.fields.map(f => (f.name, PType.canonical(f.typ))) ++ rTyp.fields.map(f => (f.name, f.typ)): _*)

    val localBufferSize = bufferSize
    val rdd = prev.rvd
      .boundary
      .mapPartitionsWithIndex { (i, ctx, it) =>
        val partRegion = ctx.partitionRegion
        val globals = globalsBc.value.readRegionValue(partRegion)
        val makeKey = {
          val f = makeKeyF(fsBc.value, i, partRegion)
          ptr: Long => {
            val keyOff = f(ctx.region, ptr, globals)
            SafeRow.read(localKeyPType, keyOff).asInstanceOf[Row]
          }
        }
        val makeAgg = { () =>
          val aggRegion = ctx.freshRegion()
          RegionValue(aggRegion, deserialize(aggRegion, initAggs))
        }

        val seqOp = {
          val f = makeSeq(fsBc.value, i, partRegion)
          (ptr: Long, agg: RegionValue) => {
            f.setAggState(agg.region, agg.offset)
            f(ctx.region, globals, ptr)
            agg.setOffset(f.getAggOffset())
          }
        }
        val serializeAndCleanupAggs = { rv: RegionValue =>
          val a = serialize(rv.region, rv.offset)
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
      }.aggregateByKey(initAggs, nPartitions.getOrElse(prev.rvd.getNumPartitions))(combOp, combOp)

    val crdd = ContextRDD.weaken(rdd).cmapPartitionsWithIndex(
      { (i, ctx, it) =>
        val region = ctx.region

        val rvb = new RegionValueBuilder()
        val partRegion = ctx.partitionRegion
        val globals = globalsBc.value.readRegionValue(partRegion)
        val annotate = makeAnnotate(fsBc.value, i, partRegion)

        it.map { case (key, aggs) =>
          rvb.set(region)
          rvb.start(newRowType)
          rvb.startStruct()
          var i = 0
          while (i < localKeyType.size) {
            rvb.addAnnotation(localKeyType.types(i), key.get(i))
            i += 1
          }

          val aggOff = deserialize(region, aggs)
          annotate.setAggState(region, aggOff)
          rvb.addAllFields(rTyp, region, annotate(region, globals))
          rvb.endStruct()
          rvb.end()
        }
      })

    prev.copy(
      typ = typ,
      rvd = RVD.coerce(ctx, RVDType(newRowType, keyType.fieldNames), crdd))
  }
}

// follows key_by non-empty key
case class TableAggregateByKey(child: TableIR, expr: IR) extends TableIR {
  require(child.typ.key.nonEmpty)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  lazy val children: IndexedSeq[BaseIR] = Array(child, expr)

  def copy(newChildren: IndexedSeq[BaseIR]): TableAggregateByKey = {
    assert(newChildren.length == 2)
    val IndexedSeq(newChild: TableIR, newExpr: IR) = newChildren
    TableAggregateByKey(newChild, newExpr)
  }

  val typ: TableType = child.typ.copy(rowType = child.typ.keyType ++ coerce[TStruct](expr.typ))

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)
    val prevRVD = prev.rvd.truncateKey(child.typ.key)
    val fsBc = ctx.fsBc

    val res = genUID()
    val extracted = agg.Extract(expr, res, Requiredness(this, ctx))

    val (_, makeInit) = ir.CompileWithAggregators[AsmFunction2RegionLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t)))),
      FastIndexedSeq(classInfo[Region], LongInfo), UnitInfo,
      extracted.init)

    val (_, makeSeq) = ir.CompileWithAggregators[AsmFunction3RegionLongLongUnit](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
        ("row", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prevRVD.rowPType)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), UnitInfo,
      extracted.seqPerElt)

    val valueIR = Let(res, extracted.results, extracted.postAggIR)
    val keyType = prevRVD.typ.kType

    val key = Ref(genUID(), keyType.virtualType)
    val value = Ref(genUID(), valueIR.typ)
    val (Some(PTypeReferenceSingleCodeType(rowType: PStruct)), makeRow) = ir.CompileWithAggregators[AsmFunction3RegionLongLongLong](ctx,
      extracted.states,
      FastIndexedSeq(("global", SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(prev.globals.t))),
        (key.name, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(keyType)))),
      FastIndexedSeq(classInfo[Region], LongInfo, LongInfo), LongInfo,
      Let(value.name, valueIR,
        InsertFields(key, typ.valueType.fieldNames.map(n => n -> GetField(value, n)))))

    assert(rowType.virtualType == typ.rowType, s"$rowType, ${ typ.rowType }")

    val localChildRowType = prevRVD.rowPType
    val keyIndices = prevRVD.typ.kFieldIdx
    val keyOrd = prevRVD.typ.kRowOrd
    val globalsBc = prev.globals.broadcast

    val newRVDType = prevRVD.typ.copy(rowType = rowType)

    val newRVD = prevRVD
      .repartition(ctx, prevRVD.partitioner.strictify)
      .boundary
      .mapPartitionsWithIndex(newRVDType) { (i, ctx, it) =>
        val partRegion = ctx.partitionRegion
        val globalsOff = globalsBc.value.readRegionValue(partRegion)

        val initialize = makeInit(fsBc.value, i, partRegion)
        val sequence = makeSeq(fsBc.value, i, partRegion)
        val newRowF = makeRow(fsBc.value, i, partRegion)

        val aggRegion = ctx.freshRegion()

        new Iterator[Long] {
          var isEnd = false
          var current: Long = 0
          val rowKey: WritableRegionValue = WritableRegionValue(keyType, ctx.freshRegion())
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  val children: IndexedSeq[BaseIR] = FastIndexedSeq(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableOrderBy = {
    val IndexedSeq(newChild) = newChildren
    TableOrderBy(newChild.asInstanceOf[TableIR], sortFields)
  }

  val typ: TableType = child.typ.copy(key = FastIndexedSeq())

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val prev = child.execute(ctx)

    val physicalKey = prev.rvd.typ.key
    if (TableOrderBy.isAlreadyOrdered(sortFields, physicalKey))
      return prev.copy(typ = typ)

    val rowType = child.typ.rowType
    val sortColIndexOrd = sortFields.map { case SortField(n, so) =>
      val i = rowType.fieldIdx(n)
      val f = rowType.fields(i)
      val fo = f.typ.ordering
      if (so == Ascending) fo else fo.reverse
    }.toArray

    val ord: Ordering[Annotation] = ExtendedOrdering.rowOrdering(sortColIndexOrd).toOrdering

    val act = implicitly[ClassTag[Annotation]]

    val codec = TypedCodecSpec(prev.rvd.rowPType, BufferSpec.wireSpec)
    val rdd = prev.rvd.keyedEncodedRDD(ctx, codec, sortFields.map(_.field)).sortBy(_._1)(ord, act)
    val (rowPType: PStruct, orderedCRDD) = codec.decodeRDD(ctx, rowType, rdd.map(_._2))
    TableValue(ctx, typ, prev.globals, RVD.unkeyed(rowPType, orderedCRDD))
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

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

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

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = child.execute(ctx).rename(globalMap, rowMap)
}

case class TableFilterIntervals(child: TableIR, intervals: IndexedSeq[Interval], keep: Boolean) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = child.rowCountUpperBound

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableFilterIntervals(newChild, intervals, keep)
  }

  override lazy val typ: TableType = child.typ

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val tv = child.execute(ctx)
    val partitioner = RVDPartitioner.union(
      tv.typ.keyType,
      intervals,
      tv.typ.keyType.size - 1)
    TableValue(ctx, tv.typ, tv.globals, tv.rvd.filterIntervals(partitioner, keep))
  }
}

case class MatrixToTableApply(child: MatrixIR, function: MatrixToTableFunction) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

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
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: TableIR) = newChildren
    TableToTableApply(newChild, function)
  }

  override lazy val typ: TableType = function.typ(child.typ)

  override def partitionCounts: Option[IndexedSeq[Long]] =
    if (function.preservesPartitionCounts) child.partitionCounts else None

  lazy val rowCountUpperBound: Option[Long] = if (function.preservesPartitionCounts) child.rowCountUpperBound else None

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    function.execute(ctx, child.execute(ctx))
  }
}

case class BlockMatrixToTableApply(
  bm: BlockMatrixIR,
  aux: IR,
  function: BlockMatrixToTableFunction) extends TableIR {

  override lazy val children: IndexedSeq[BaseIR] = Array(bm, aux)

  lazy val rowCountUpperBound: Option[Long] = None

  override def copy(newChildren: IndexedSeq[BaseIR]): TableIR =
    BlockMatrixToTableApply(
      newChildren(0).asInstanceOf[BlockMatrixIR],
      newChildren(1).asInstanceOf[IR],
      function)

  override lazy val typ: TableType = function.typ(bm.typ, aux.typ)

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    val b = bm.execute(ctx)
    val a = CompileAndEvaluate[Any](ctx, aux, optimize = false)
    function.execute(ctx, b, a)
  }
}

case class BlockMatrixToTable(child: BlockMatrixIR) extends TableIR {
  lazy val children: IndexedSeq[BaseIR] = Array(child)

  lazy val rowCountUpperBound: Option[Long] = None

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newChild: BlockMatrixIR) = newChildren
    BlockMatrixToTable(newChild)
  }

  override val typ: TableType = {
    val rvType = TStruct("i" -> TInt64, "j" -> TInt64, "entry" -> TFloat64)
    TableType(rvType, Array[String](), TStruct.empty)
  }

  protected[ir] override def execute(ctx: ExecuteContext): TableValue = {
    child.execute(ctx).entriesTable(ctx)
  }
}

case class RelationalLetTable(name: String, value: IR, body: TableIR) extends TableIR {
  def typ: TableType = body.typ

  lazy val rowCountUpperBound: Option[Long] = body.rowCountUpperBound

  def children: IndexedSeq[BaseIR] = Array(value, body)

  def copy(newChildren: IndexedSeq[BaseIR]): TableIR = {
    val IndexedSeq(newValue: IR, newBody: TableIR) = newChildren
    RelationalLetTable(name, newValue, newBody)
  }
}
